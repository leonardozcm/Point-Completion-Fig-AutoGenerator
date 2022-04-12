import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('..')
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

def performance_Cal(model_type, dataset):

    if dataset is 'ShapeNetCompletion' :
        root_path= os.path.join('../../dataset', dataset, model_type, 'test')
        score_CD = 0
        num_pointcloud = 0
        CD_category = np.zeros(8)
        i_category = 0
        for category in os.listdir(root_path):
            path= os.path.join(root_path, category)
            pcd_path = path + "/pcds/dense/"
            num_category = 0
            for file_dir in os.listdir(pcd_path):
                file_pcd = pcd_path + file_dir
                dense_cloud = IO.get(file_pcd).astype(np.float32)
                dense_cloud = np.expand_dims(dense_cloud, axis=0)
                dense_cloud = torch.Tensor(dense_cloud)
                dense_cloud = dense_cloud.cuda()

                gt_path = file_pcd.replace("dense", "gt")
                gt_cloud = IO.get(gt_path).astype(np.float32)
                gt_cloud = np.expand_dims(gt_cloud, axis=0)
                gt_cloud = torch.Tensor(gt_cloud)
                gt_cloud = gt_cloud.cuda()
                score_CD_indiviual = chamfer_dist(dense_cloud, gt_cloud)
                score_CD = score_CD + score_CD_indiviual
                CD_category[i_category] = CD_category[i_category] + score_CD_indiviual
                num_pointcloud = num_pointcloud + 1
                num_category = num_category + 1
                print('ID:', num_pointcloud, 'CD:', score_CD.item()*10000 / num_pointcloud)
            i_category = i_category + 1
        print("Airplane:", CD_category[0] * 10000 / num_category)
        print("Cabinet:", CD_category[1] * 10000 / num_category)
        print("Car:", CD_category[2] * 10000 / num_category)
        print("Chair:", CD_category[3] * 10000 / num_category)
        print("Lamp:", CD_category[4] * 10000 / num_category)
        print("Sofa:", CD_category[5] * 10000 / num_category)
        print("Table:", CD_category[6] * 10000 / num_category)
        print("Vessel:", CD_category[7] * 10000 / num_category)

    if dataset is 'Completion3D' :
        root_path = os.path.join('../../dataset', dataset, model_type, 'val')
        score_CD = 0
        num_pointcloud = 0
        CD_category = np.zeros(8)
        i_category = 0
        for category in os.listdir(root_path):
            path = os.path.join(root_path, category)
            pcd_path = path + "/pcds/dense/"
            num_category = 0
            for file_dir in os.listdir(pcd_path):
                file_pcd = pcd_path + file_dir
                dense_cloud = IO.get(file_pcd).astype(np.float32)
                dense_cloud = np.expand_dims(dense_cloud, axis=0)
                dense_cloud = torch.Tensor(dense_cloud)
                dense_cloud = dense_cloud.cuda()

                gt_path = file_pcd.replace("dense", "gt")
                gt_cloud = IO.get(gt_path).astype(np.float32)
                gt_cloud = np.expand_dims(gt_cloud, axis=0)
                gt_cloud = torch.Tensor(gt_cloud)
                gt_cloud = gt_cloud.cuda()
                score_CD_indiviual = chamfer_dist(dense_cloud, gt_cloud)
                score_CD = score_CD + score_CD_indiviual
                CD_category[i_category] = CD_category[i_category] + score_CD_indiviual
                num_pointcloud = num_pointcloud + 1
                num_category = num_category + 1
                print('ID:', num_pointcloud, 'CD:', score_CD.item() * 10000 / num_pointcloud)
            i_category = i_category + 1
        print("Airplane:", CD_category[0] * 10000 / num_category)
        print("Cabinet:", CD_category[1] * 10000 / num_category)
        print("Car:", CD_category[2] * 10000 / num_category)
        print("Chair:", CD_category[3] * 10000 / num_category)
        print("Lamp:", CD_category[4] * 10000 / num_category)
        print("Sofa:", CD_category[5] * 10000 / num_category)
        print("Table:", CD_category[6] * 10000 / num_category)
        print("Vessel:", CD_category[7] * 10000 / num_category)


if __name__ == "__main__":

    scale = 128

    model_type = 'PRNet_Voxel'

    dataset = 'ShapeNetCompletion'
    # dataset = 'Completion3D'

    performance_Cal(model_type, dataset)