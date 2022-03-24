#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from torch import nn, einsum
from SiaTrans.utils import MLP_Res, grouping_operation, query_knn


class SkipTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2,
                             hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped,
                            pos_flipped, include_self=include_self)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - \
            grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #

        agg = einsum('b c i j, b c i j -> b c i',
                     attention, value)  # b, dim, n
        y = self.conv_end(agg)

        return y + identity


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channel):
        super(Self_Attn, self).__init__()

        self.query_conv = nn.Conv1d(
            in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.key_conv = nn.Conv1d(
            in_channels=in_channel, out_channels=in_channel//8, kernel_size=1)
        self.value_conv = nn.Conv1d(
            in_channels=in_channel, out_channels=in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X n_points)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, npoints = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, npoints).permute(0, 2, 1)  # B X CX(N)
        # print(proj_query.size())
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, npoints)  # B X C x (*W*H)
        # print(proj_key.size())
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, npoints)  # B X C X N
        # print(proj_value.size())

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, npoints)

        out = self.gamma*out + x
        return out


# bs = 8
# input = torch.randn((bs, 3, 512))
# feat_global = torch.randn((bs, 32, 512))
# K_rev = torch.randn((bs, 128, 512))


# ps = nn.ConvTranspose1d(32, 128, 4, 4, bias=False)
# res = ps(feat_global)

# up_sampler = nn.Upsample(scale_factor=4)
# H = torch.randn((bs, 128, 512))
# H_up = up_sampler(H)
# print(res.size())
# print(H_up.size())

class PointNonLocalCell(nn.Module):
    """Input
        feature: (batch_size, ndataset, channel) Torch tensor
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, channel)
    """

    def __init__(self, feature_channel, in_channel, mlps):
        super(PointNonLocalCell, self).__init__()
        self.bottleneck_channel = mlps[0]
        self.transformed_feature = nn.Conv2d(
            feature_channel, self.bottleneck_channel * 2, 1)
        self.tf_bn = nn.BatchNorm2d(self.bottleneck_channel*2)
        self.transformed_new_point = nn.Conv2d(
            in_channel, self.bottleneck_channel, 1)
        self.tnp_bn = nn.BatchNorm2d(self.bottleneck_channel)
        self.new_nonlocal_point = nn.Conv2d(
            self.bottleneck_channel, mlps[-1], 1)
        self.nnp_bn = nn.BatchNorm2d(mlps[-1])

    def forward(self, feature, new_point):
        B, P, C, S = new_point.shape
        FB, FC, FD = feature.shape
        feature = feature.view(FB, FC, 1, FD).permute(0, 1, 3, 2)
        new_point = new_point.permute(0, 2, 1, 3)

        # B, P, S, C = new_point.shape
        # FB, FD, FC = feature.shape
        # feature = feature.view(FB, FD, 1, FC).permute(0,3,1,2) FB, FC, FD, 1
        # new_point = new_point.permute(0,3,1,2) B, C, P, S

        transformed_feature = self.tf_bn(
            self.transformed_feature(feature)).permute(0, 2, 3, 1)
        transformed_new_point = self.tnp_bn(
            self.transformed_new_point(new_point)).permute(0, 2, 3, 1)
        transformed_new_point = torch.reshape(
            transformed_new_point, (B, P*S, self.bottleneck_channel))
        transformed_feature1 = torch.squeeze(
            transformed_feature[:, :, :, :self.bottleneck_channel], 2)
        transformed_feature2 = torch.squeeze(
            transformed_feature[:, :, :, self.bottleneck_channel:], 2)

        attention_map = torch.matmul(
            transformed_new_point, transformed_feature1.transpose(1, 2))  # mode = 'dot'
        attention_map = attention_map / np.sqrt(self.bottleneck_channel)
        attention_map = F.softmax(attention_map, -1)

        new_nonlocal_point = torch.matmul(attention_map, transformed_feature2)
        new_nonlocal_point = torch.reshape(
            new_nonlocal_point, (B, P, S, self.bottleneck_channel)).permute(0, 3, 1, 2)
        new_nonlocal_point = self.nnp_bn(self.new_nonlocal_point(
            new_nonlocal_point)).permute(0, 2, 1, 3)
        # print(new_nonlocal_point.size())
        new_nonlocal_point = torch.squeeze(new_nonlocal_point, 1)

        return new_nonlocal_point


# bs = 16
# new_feature = torch.randn((bs, 128, 512))
# new_points = torch.randn((bs, 1, 128, 512))
# mlp = [128, 128, 256]
# PNL = PointNonLocalCell(128, 128, [max(32, 128+3//2), 128])
# new_nonlocal_point = PNL(new_feature, new_points)
# print(new_nonlocal_point.size())
