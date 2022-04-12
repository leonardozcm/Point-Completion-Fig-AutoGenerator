import sys
import os
from utils.file_parsing import file_parsing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# from torchvision import utils as vtils
# import time
# import logging
# from models.prnet import PRNet
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from utils.io import IO
import numpy as np
import torch
# import cv2
from extensions.chamfer_dist import ChamferDistance

chamfer_dist = ChamferDistance()

def consistency(path):
    pcd_path = os.path.join(path, "pcds", "dense")
    parsing_list = file_parsing(pcd_path)
    score_CD = 0
    num = 0
    for i in range(len(parsing_list)):
        car_frames = parsing_list[i]
        for j in range(len(car_frames)-1):
            forward_path = os.path.join(pcd_path, car_frames[j])
            forward_cloud = IO.get(forward_path).astype(np.float32)
            forward_cloud = np.expand_dims(forward_cloud, axis=0)
            forward_cloud = torch.Tensor(forward_cloud)
            forward_cloud = forward_cloud.cuda()

            next_path = os.path.join(pcd_path, car_frames[j+1])
            next_cloud = IO.get(next_path).astype(np.float32)
            next_cloud = np.expand_dims(next_cloud, axis=0)
            next_cloud = torch.Tensor(next_cloud)
            next_cloud = next_cloud.cuda()

            score_CD_indiviual = chamfer_dist(forward_cloud, next_cloud)
            num = num + 1
            score_CD += score_CD_indiviual
    print("Consistency: ", score_CD.item()/num)


if __name__ == "__main__":
    consistency("../../../dataset/KITTI")