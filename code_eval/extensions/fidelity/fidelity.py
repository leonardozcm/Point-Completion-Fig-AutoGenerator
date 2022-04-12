import torch
import os
from utils.io import IO
import numpy as np
from utils.file_parsing import file_parsing

def square_distance(src, dst, normalised=False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points , [B, N, C]
        dst: target points , [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    # 得到两个点云的shaoe
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # 这里先计算  -2xy
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # 忽视这里的判断，主要是对距离的计算添加一个常数项，用作正则化
    if (normalised):
        dist += 2
    # 这里计算 x^2 和 y^2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    # 使用 clamp函数将距离规范化到固定的区间
    dist = dist.sqrt()
    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def fidelity(path):
    input_path = os.path.join(path, "partial", "02958343")
    output_path = os.path.join(path, "pcds", "dense")
    parsing_list = file_parsing(output_path)
    score = 0
    num = 0
    for i in range(len(parsing_list)):
        car_frames = parsing_list[i]
        for j in range(len(car_frames)):
            input_pcd_path = os.path.join(input_path, car_frames[j].split(".")[0], car_frames[j])
            output_pcd_path = os.path.join(output_path, car_frames[j])
            input_cloud = IO.get(input_pcd_path).astype(np.float32)
            input_cloud = np.expand_dims(input_cloud, axis=0)
            input_cloud = torch.Tensor(input_cloud)
            input_cloud = input_cloud.cuda()

            output_cloud = IO.get(output_pcd_path).astype(np.float32)
            output_cloud = np.expand_dims(output_cloud, axis=0)
            output_cloud = torch.Tensor(output_cloud)
            output_cloud = output_cloud.cuda()\

            square_dist = square_distance(input_cloud, output_cloud, normalised=False)
            score_indiviual = 0
            for k in range(len(square_dist[0])):
                min_score = min(square_dist[0][k])
                print(min_score)
                score_indiviual = score_indiviual + min_score
            score_indiviual = score_indiviual/len(square_dist[0])
            num = num + 1
            score = score + score_indiviual
    score = score / num
    return score



if __name__ == "__main__":
    print(fidelity("../../../dataset/KITTI/"))