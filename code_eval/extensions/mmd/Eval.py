import sys
import os
from utils.io import IO
import numpy as np
import torch
from utils.shapenet_pre_handle import pre_handle
from extensions.chamfer_dist import ChamferDistance
from utils.file_parsing import file_parsing

chamfer_dist = ChamferDistance()

def MMD_Cal(shapenet_list):
    shapenet_MMD = []
    for folders in shapenet_list:
        folder_MMD = []
        for couples in folders:
            forward_cloud = IO.get(couples[0]).astype(np.float32)
            forward_cloud = np.expand_dims(forward_cloud, axis=0)
            forward_cloud = torch.Tensor(forward_cloud)
            forward_cloud = forward_cloud.cuda()

            next_cloud = IO.get(couples[1]).astype(np.float32)
            next_cloud = np.expand_dims(next_cloud, axis=0)
            next_cloud = torch.Tensor(next_cloud)
            next_cloud = next_cloud.cuda()

            score_MMD = chamfer_dist(forward_cloud, next_cloud).item()
            folder_MMD.append(score_MMD)
        shapenet_MMD.append(folder_MMD)
    avg_list = []
    for folders in shapenet_MMD:
        avg = np.mean(folders)
        avg_list.append(avg)
    print(avg_list)
    avg = np.mean(avg_list)
    return avg


if __name__ == "__main__":
    # shapenet_list = pre_handle("../../../dataset/ShapeNetCompletion/PRNet_Voxel/test")
    # print(shapenet_list)
    path = "G:\zhangjunwei\Eval_KITTI\dataset\KITTI\pcds\dense"
    list = file_parsing(path)
    MMD = MMD_Cal(list)
    print(MMD)
