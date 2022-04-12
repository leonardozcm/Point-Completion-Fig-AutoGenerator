import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
from ASFMNet.modelutils import fps_subsample

from extensions.emd.emd_module import emdModule
emd = emdModule()
import numpy as np

def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def f1_score(p1,p2,th=0.0001):
    dist1, dist2, _, _ = chamfer_dist(p1, p2)
    dist1=dist1.squeeze()
    dist2=dist2.squeeze()
    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0


def emd_loss(p1,p2):
    dis, _ = emd(p1, p2, 0.005, 50)
    return np.sqrt(dis.cpu()).mean()

def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

def get_loss(pcds_pred, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)* 1e3
    # print(Pc.size(), gt_c.size())
    cd1 = CD(P1, gt_1)* 1e3
    # print(P1.size(), gt_1.size())
    cd2 = CD(P2, gt_2)* 1e3
    # print(P2.size(), gt_2.size())
    cd3 = CD(P3, gt)* 1e3
    # print(P3.size(), gt.size())

    loss_all = (cdc + cd1 + cd2 + cd3) * 1e3
    losses = (cdc, cd1, cd2, cd3)
    return loss_all, losses
