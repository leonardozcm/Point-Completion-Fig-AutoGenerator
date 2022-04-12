# type: ignore

import matplotlib
from tomlkit import key

from ASFMNet.modelutils import fps_subsample
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
        # print(pred[0].size()) 
        cd_256 = chamfer_sqrt(pred[0],fps_subsample(gt,256))
        cd_16384 = chamfer_sqrt(pred[1], gt)
        return (cd_256.item() * 1e3, cd_16384.item()*1e3)
    
    def getImg(self, partial):
        pred = self.model(partial)
        ptcloud_256 = pred[0].squeeze().cpu().numpy()
        ptcloud_img_256 = utils.helpers.get_ptcloud_img(ptcloud_256)
        ptcloud_16384 = pred[1].squeeze().cpu().numpy()
        ptcloud_img_16384 = utils.helpers.get_ptcloud_img(ptcloud_16384)
        return ptcloud_img_256,ptcloud_img_16384

    def eval(self):
        self.model.eval()

def crop_img(img, factor=0.25):
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

                imgs = model.getImg(partial)
                dir_name = "visualization/"+sub_dir_name+"/"+id[0] + "/"+ model.name
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                
                img_croped_256 = crop_img(imgs[0])
                cv2.imwrite(dir_name+"/"+save_name.split(".")[0]+"_256.png",img_croped_256)
                img_list.append((save_name, img_croped_256))

                img_croped_16384 = crop_img(imgs[1])
                cv2.imwrite(dir_name+"/"+save_name.split(".")[0]+"_16384.png",img_croped_16384)
    return img_list

def concatImg(imgLists):
    himg = []
    for model_instance in imgLists:
        himg.append(cv2.hconcat([x[1] for x in model_instance[1]]))

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
                    img = utils.helpers.get_ptcloud_img(pcd.squeeze().cpu().numpy())
                    dir_name = "visualization/"+sub_dir_name+"/"+id[0] + "/"+ k
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    img_croped = crop_img(img)
                    cv2.imwrite(dir_name+"/"+save_name+".png",img_croped)
                    ls.append((save_name, img_croped))
    
    return [('partial_cloud',data_keys['partial_cloud']), ('gtcloud',data_keys['gtcloud'])]


def select_outperforms(threshold=1.0):
    # taxonomy_map = {
    #     "02691156":[],
    #     "02933112":[],
    #     "02958343":[],
    #     "03001627":[],
    #     "03636649":[],
    #     "04256520":[],
    #     "04379243":[],
    #     "04530566":[]
    # }

    file_name = []
    for file, pair in result.items():
        if pair["SnowflakeNet"][0] > pair["ASFMNet"][0] and pair["ASFMNet"][1] > pair["SnowflakeNet"][1]:
            file_name.append((file,  pair["SnowflakeNet"][0]-pair["ASFMNet"][0], pair["ASFMNet"][0]))
    

    file_name=sorted(file_name,key=lambda x:x[1],reverse=True)[:10]
    
    
    return file_name

sub_dir_name = "asfm_runner"

if os.path.exists('visualization/'+sub_dir_name):
    import shutil
    shutil.rmtree('visualization/'+sub_dir_name)  
os.mkdir('visualization/'+sub_dir_name) 

result = {}
dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
    utils.data_loaders.DatasetSubset.TEST),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=utils.data_loaders.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)

from Snowflake.model import SnowflakeNet
from ASFMNet.SApcn import ASFM


Model_list = [
    # CandiModel("GRNet", loadParallelModel(GRNet(),"checkpoint/grnet.pth",keyname='grnet')),
    # CandiModel("PCN", pcn_model),
    # CandiModel("VRCNet", loadParallelModel(Vrcnet(),"checkpoint/pretrained_vrcnet_2048.pth",keyname='net_state_dict', parallel=False)),
    # CandiModel("PMPNet", loadParallelModel(PMPNet(),"checkpoint/pmpnet.pth") ),
    CandiModel("ASFMNet", loadParallelModel(ASFM(step_ratio=16,rt_coarse=True),"checkpoint/asfm16384.pth",)),
    CandiModel("SnowflakeNet", loadParallelModel(SnowflakeNet(up_factors=[4,8],rt_coarse=True),"checkpoint/snowflakenet.pth")),
    # CandiModel("ASFMNet", loadParallelModel(SiaTrans(up_factors=[4,8]),"checkpoint/ASFMNet.pth")),
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
# img_dict.insert(0,head_tail[0])
# img_dict.append(head_tail[1])
# cv2.imwrite("visualization/concat.png",concatImg(img_dict))
