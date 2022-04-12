import os


def pre_handle(path):
    shapenet_list = []
    for folder in os.listdir(path):
        dense_path = path + "/" + folder + "/pcds/dense/"
        gt_path = path + "/" + folder + "/pcds/gt/"
        item_list = []
        for pcd in os.listdir(dense_path):
            item_list.append([dense_path + pcd,gt_path + pcd])
        shapenet_list.append(item_list)
    return shapenet_list


if __name__ == "__main__":
    path = "../../dataset/ShapeNetCompletion/PRNet_Voxel/test"
    shape_net_list = pre_handle(path)
    print(shape_net_list)
