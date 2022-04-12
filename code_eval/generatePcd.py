import torch
import open3d
import os
import numpy as np
from utils.file_parsing import file_parsing
from utils.io import IO
from tqdm import tqdm
import sys
sys.path.append("../")
from SiaTrans.model import SiaTrans

def write_tensor2pcd(cloud_tensor, name):
    cloud_tensor=cloud_tensor.squeeze(dim=0)
    cloud_tensor = cloud_tensor.to(torch.device('cpu')).squeeze().detach().numpy()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(cloud_tensor)
    open3d.io.write_point_cloud(name, pcd, write_ascii=True)

def loadParallelModel(model, path, subkey=True, keyname='model', parallel=True):
    checkpoint = torch.load(path)
    if parallel:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model=model.cuda()
    if subkey:
        print(model.load_state_dict(checkpoint[keyname]))
    else:
        print(model.load_state_dict(checkpoint))
    return model

def run(path):
    pcd_path = os.path.join(path, "partial", "02958343")
    parsing_list = file_parsing(pcd_path)
    print(parsing_list)
    model=loadParallelModel(SiaTrans(up_factors=[4,8]),"../checkpoint/ours.pth")
    for i in tqdm(range(len(parsing_list))):
        car_frames = parsing_list[i]
        for j in range(len(car_frames)-1):
            forward_path = os.path.join(pcd_path, car_frames[j])
            forward_cloud = IO.get(forward_path).astype(np.float32)
            forward_cloud = np.expand_dims(forward_cloud, axis=0)
            forward_cloud = torch.Tensor(forward_cloud).cuda()
            # print(forward_cloud.size())

            output = model(forward_cloud)
            write_tensor2pcd(output, os.path.join(path, "pcds", "dense", car_frames[j]+".pcd"))


if __name__ == "__main__":
    run("./dataset")