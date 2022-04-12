# type: ignore

from tkinter.tix import Tree
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
from utils.loss_utils import get_loss


class CandiModel(object):
    def __init__(self, name, model=None) -> None:
        self.name = name
        self.model = model
    
    def __call__(self, data, gt):
        pred = self.model(data)
        # print(pred[0].size())
        _, losses = get_loss(pred,gt)
        return losses
    
    def getImg(self, partial):
        pred = self.model(partial)
        imgs = []
        for p in pred:
            ptcloud_256 = p.squeeze().cpu().numpy()
            ptcloud_img_256 = utils.helpers.get_ptcloud_img(ptcloud_256)
            imgs.append(ptcloud_img_256)           
        return imgs

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
    resolutions = [256,512,2048,16384]


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
                
                for i in range(4):
                
                    img_croped_256 = crop_img(imgs[i])
                    cv2.imwrite(dir_name+"/"+save_name.split(".")[0]+"_"+str(resolutions[i])+".png",img_croped_256)

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
        cd_256,cd_512,cd_2048,cd_16384 = pair["SnowflakeNet"]
        # if cd_256-cd_512>7.4671 and cd_2048-cd_512>7.5245 and cd_16384-cd_2048 > 5.2214:
        #     file_name.append((file,  cd_256-cd_16384, cd_256))
        if cd_16384 > 13:
            file_name.append((file, cd_16384, cd_256))          
    

    file_name=sorted(file_name,key=lambda x:x[2], reverse=False)[:10]
    
    
    return file_name

sub_dir_name = "snowflake_runner"

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


model = CandiModel("SnowflakeNet", loadParallelModel(SnowflakeNet(up_factors=[4,8],rt_coarse=True),"checkpoint/snowflakenet.pth"))
test_net(test_data_loader, model)

file_select = select_outperforms(1.1)
for x in file_select:
    print(x[2])

generateImg(test_data_loader, model,file_select)
head_tail = writeinputandgt(test_data_loader,file_select)
# img_dict.insert(0,head_tail[0])
# img_dict.append(head_tail[1])
# cv2.imwrite("visualization/concat.png",concatImg(img_dict))
