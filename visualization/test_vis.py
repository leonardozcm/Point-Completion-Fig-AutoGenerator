root = "./visualization/"
pcd_path = root+ "output.pcd"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import cv2
import open3d as o3d
from utils.helpers import get_ptcloud_img
from utils.io import IO

def get_ptcloud_img_fromfile(path):
    data = IO.get(path).astype(np.float32)
    img = get_ptcloud_img(data)
    return img



if __name__ == "__main__":


    data = IO.get(pcd_path).astype(np.float32)
    img = get_ptcloud_img(data)
    cv2.imwrite(root+"cam.jpg",img)
    cv2.waitKey(0)

