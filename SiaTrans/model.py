#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
import torch.nn as nn
from SiaTrans.utils import MLP_Res, MLP_CONV, fps_subsample, Transformer
from SiaTrans.skip_transformer import SkipTransformer, PointNonLocalCell

from SiaTrans.utils import PointNet_SA_Module_KNN
# from snowmodels.pointconv_util import PointConvDensitySetAbstraction as PointNet_SA_Module_KNN


class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=512):
        """Encoder that encodes information of partial point cloud
        """
        super(FeatureExtractor, self).__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(
            512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=False, bandwidth=0.1)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(
            128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=False, bandwidth=0.2)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(
            1, None, 256, [512, out_dim], group_all=True, if_bn=False, bandwidth=0.4)

    def forward(self, point_cloud):
        """
        Args:
            point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points = self.sa_module_1(
            l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)

        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points = self.sa_module_2(
            l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 128)
        # print("l2:", l2_xyz.size(), l2_points.size())
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(
            l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128,
                             hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        # (b, 128, 256)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1, self_attention=False):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 +
                              dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        if self_attention:
            self.self_attention = PointNonLocalCell(128, 128, [64, 128])

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(
            32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(
            in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)
        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[
                                0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(
            pcd_prev, K_prev if K_prev is not None else Q, Q)

        if hasattr(self, "self_attention"):
            K_global = self.self_attention(Q, torch.unsqueeze(Q,1))
            H+=K_global

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / \
            self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr

class SiaTrans(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=[4, 8],c3d=False):
        super(SiaTrans, self).__init__()

        self.encoder = FeatureExtractor()
        self.decoder = SeedGenerator()
        self.num_p0 = num_p0
        self.c3d=c3d

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            self_atten = (i != 2)
            uppers.append(
                SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius, self_attention=self_atten))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, partial, gt=None):
        '''
         x(bs,3.N)

         returns:
         v(bs, 1024)
         y_coarse(bs, 4096, 3)
         y_fine(bs, 4096, 3)
        '''
        if gt is not None:
            x = gt
        else:
            x = partial
        x = x.permute(0, 2, 1).contiguous()
        feat = self.encoder(x)
        arr_pcd = []
        pcd = self.decoder(feat).permute(
            0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)

        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, K_prev = upper(pcd, feat, K_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        # print("y_fine: ", y_fine.shape)
        if self.c3d:
            return fps_subsample(arr_pcd[-1],2048)
        else:
            return arr_pcd[-1]

if __name__ == "__main__":
    pcs = torch.rand(16, 2048,3).cuda()
    ae = SiaTrans(up_factors=[4, 8]).cuda()
    # ae_decoder = ASFMDecoder().cuda()

    arr_pcd = ae(pcs)

    print("1:", arr_pcd.size())

    checkpoint = torch.load("checkpoint/ours.pth")
    ae_pal = torch.nn.DataParallel(ae).cuda()
    print(ae_pal.load_state_dict(checkpoint['model'], strict=False))
