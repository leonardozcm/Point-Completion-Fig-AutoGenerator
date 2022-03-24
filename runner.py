import matplotlib
matplotlib.use("Agg")
import torch
from utils.loss_utils import chamfer_sqrt
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
import os
import cv2
from config_pcn import cfg


class CandiModel(object):
    def __init__(self, name, model=None) -> None:
        self.name = name
        self.model = model
    
    def __call__(self, data, gt):
        pred = self.model(data)
        cd = chamfer_sqrt(pred, gt)
        return cd.item() * 1e3
    
    def getImg(self, partial):
        pred = self.model(partial)
        ptcloud = pred.squeeze().cpu().numpy()
        ptcloud_img = utils.helpers.get_ptcloud_img(ptcloud)
        return ptcloud_img


    def eval(self):
        self.model.eval()

def loadParallelModel(model, path, subkey=True):
    checkpoint = torch.load(path)
    model = torch.nn.DataParallel(model).cuda()
    if subkey:
        print(model.load_state_dict(checkpoint['model']))
    else:
        print(model.load_state_dict(checkpoint))
    return model

def test_net(test_data_loader=None, model=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    model.eval()
    with tqdm(test_data_loader) as t:
        for _, (_, file_path, data) in enumerate(t):
            file_path = file_path[0]
            if file_path not in result.keys():
                result[file_path]={}

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                # b, n, 3

                cd = model(partial.contiguous(), gt)

                result[file_path][model.name]=cd


def generateImg(test_data_loader=None, model=None, files=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    model.eval()


    with tqdm(test_data_loader) as t:
        for _, (id, file_path, data) in enumerate(t):
            file_path = file_path[0]
            if file_path not in [k[0] for k in files]:
                continue
            save_name = file_path.split("/")[-1]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                # b, n, 3

                img = model.getImg(partial)
                dir_name = "visualization/imgs"+id[0] + "/"+ model.name
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                cv2.imwrite(dir_name+"/"+save_name+".png", img)


def select_outperforms():
    file_name = []
    for file, pair in result.items():
        file_name.append((file, pair["SnowflakeNet"]-pair["Ours"]))
    
    file_name= sorted(file_name, key=lambda x:x[1], reverse=True)

    # for x in file_name:
    #   print(x[1])
    
    return file_name[:10]

if os.path.exists('visualization/imgs'):
    import shutil
    shutil.rmtree('visualization/imgs')  
os.mkdir('visualization/imgs') 

result = {}
dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
    utils.data_loaders.DatasetSubset.TEST),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=utils.data_loaders.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)
from SiaTrans.model import SiaTrans
from Snowflake.model import SnowflakeNet

Model_list = [
    # CandiModel("GRNet", "checkpoint/grnet.pth", ),
    # CandiModel("PCN", "checkpoint/pcn.pth", ),
    # CandiModel("PMPNet", "checkpoint/pmpnet.pth", ),
    CandiModel("SnowflakeNet", loadParallelModel(SnowflakeNet(up_factors=[4,8]),"checkpoint/snowflakenet.pth") ),
    CandiModel("Ours", loadParallelModel(SiaTrans(up_factors=[4,8]),"checkpoint/ours.pth")),
]

for model in Model_list:
    test_net(test_data_loader, model)



file_select = select_outperforms()


print(file_select)

for model in Model_list:
    generateImg(test_data_loader, model,file_select)
