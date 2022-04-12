import os


def frame(input):
    return int(input.split(".")[0].split("_")[1])

def file_parsing(path):
    pcds = sorted(os.listdir(path), key=frame)
    car_num = 0
    result = []
    # 最多有多少个汽车
    for file_name in pcds:
        if(int(file_name.split(".")[0].split("_")[3])>car_num):
            car_num = int(file_name.split(".")[0].split("_")[3])
    # 根据汽车数添加空列表
    for i in range((car_num + 1)):
        result.append([])
    # 将同一个汽车的不同帧放在同一个列表中
    for file_name in pcds:
        car = int(file_name.split(".")[0].split("_")[3])
        result[car].append(file_name)
    # print(result)
    return result



if __name__ == "__main__":
    path = "../../dataset/KITTI/pcds/dense/"
    list = file_parsing(path)
    print(list)