# type: ignore

import matplotlib
from tomlkit import key

from ASFMNet.modelutils import fps_subsample
matplotlib.use("Agg")
import torch
from utils.loss_utils import chamfer
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
import os
import cv2
from config_c3d import cfg


class CandiModel(object):
    def __init__(self, name, model=None) -> None:
        self.name = name
        self.model = model
    
    def __call__(self, data, gt):
        pred = self.model(data)
        cd = chamfer(pred, gt)
        return cd.item() * 1e4
    
    def getImg(self, partial):
        pred = self.model(partial)
        ptcloud = pred.squeeze().cpu().numpy()
        ptcloud_img = utils.helpers.plot_single_pcd(ptcloud)

        return ptcloud_img

    def eval(self):
        self.model.eval()

def crop_img(img, factor=0.25):
    # return img 
    w,_,_=img.shape
    crop_len = int(factor*w)
    img_crop = img[crop_len: w-crop_len, crop_len:w-crop_len,:]
    return img_crop

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

    img_list = []


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
                dir_name = "visualization/imgs/"+id[0] + "/"+ model.name
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                img_croped = crop_img(img)
                cv2.imwrite(dir_name+"/"+save_name+".png",img_croped)
                img_list.append((save_name, img_croped))
    return img_list

def concatImg(imgLists):
    himg = []
    for model_instance in imgLists:
        img = cv2.hconcat([x[1] for x in model_instance[1]])
        # print(img.shape)
        himg.append(img)

    return cv2.vconcat(himg)


def writeinputandgt(test_data_loader=None, files=None):
    partial_list = []
    gtcloud_list = []
    data_keys = {'partial_cloud':partial_list, 'gtcloud':gtcloud_list}

    with tqdm(test_data_loader) as t:
        for _, (id, file_path, data) in enumerate(t):
            file_path = file_path[0]
            if file_path not in [k[0] for k in files]:
                continue
            save_name = file_path.split("/")[-1]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                
                
                for k, ls in data_keys.items():
                    pcd = data[k]
                    pcd = fps_subsample(pcd,2048)
                    img = utils.helpers.plot_single_pcd(pcd.squeeze().cpu().numpy())
                    dir_name = "visualization/imgs/"+id[0] + "/"+ k
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    img_croped = crop_img(img)
                    cv2.imwrite(dir_name+"/"+save_name+".png",img_croped)
                    ls.append((save_name, img_croped))
    
    return [('partial_cloud',data_keys['partial_cloud']), ('gtcloud',data_keys['gtcloud'])]


def select_outperforms(threshold=1.0):
    taxonomy_map = {
        "02691156":[],
        "02933112":[],
        "02958343":[],
        "03001627":[],
        "03636649":[],
        "04256520":[],
        "04379243":[],
        "04530566":[]
    }
    taxes = list(taxonomy_map.keys())

    for file, pair in result.items():
        file_tax = file.split("/")[-2]
        if pair["SnowflakeNet"]-pair["Ours"] > threshold:
            taxonomy_map[file_tax].append((file,  pair["SnowflakeNet"]-pair["Ours"], pair["Ours"]))
    
    file_name = []
    for k,v in taxonomy_map.items():
        file_name.extend(sorted(v, key=lambda x:x[2])[:5])

    # for x in file_name:
    #   print(x[1])
    
    
    return file_name

if os.path.exists('visualization/imgs'):
    import shutil
    shutil.rmtree('visualization/imgs')  
os.mkdir('visualization/imgs') 

result = {}
dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
    utils.data_loaders.DatasetSubset.VAL),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=utils.data_loaders.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)
from SiaTrans.model import SiaTrans
from Snowflake.model import SnowflakeNet
from GRNet.grnet import GRNet
# from PMPNet.model import Model as PMPNet
from PCN.pcn import PCN
from VRCNet.vrcet import Model as Vrcnet
from ASFMNet.SApcn import ASFM

# load PCN
pcn_model = PCN(num_dense=16384, latent_dim=1024, grid_size=4,c3d=True).cuda()
print(pcn_model.load_state_dict(torch.load("checkpoint/best_l1_cd.pth")))


Model_list = [
    CandiModel("GRNet", loadParallelModel(GRNet(c3d=True),"checkpoint/grnet.pth",keyname='grnet')),
    CandiModel("PCN", pcn_model),
    CandiModel("VRCNet", loadParallelModel(Vrcnet(),"checkpoint/pretrained_vrcnet_2048.pth",keyname='net_state_dict', parallel=False)),
    CandiModel("ASFM", loadParallelModel(ASFM(c3d=True),"checkpoint/asfm.pth")),
    CandiModel("SnowflakeNet", loadParallelModel(SnowflakeNet(up_factors=[2,2]),"checkpoint/c3d/snowflakenet.pth") ),
    CandiModel("Ours", loadParallelModel(SiaTrans(up_factors=[2,2]),"checkpoint/c3d/ours.pth")),
]

for model in Model_list[-2:]:
    test_net(test_data_loader, model)

file_select = select_outperforms(1.1)
for x in file_select:
    print(x[1])

img_dict = []

for model in Model_list:
    img_dict.append((model.name,generateImg(test_data_loader, model,file_select)))

head_tail = writeinputandgt(test_data_loader,file_select)
img_dict.insert(0,head_tail[0])
img_dict.append(head_tail[1])
cv2.imwrite("visualization/concat.png",concatImg(img_dict))
