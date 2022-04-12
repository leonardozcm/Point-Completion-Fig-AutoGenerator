# -*- coding: utf-8 -*-
# @Author: XP
# type: ignore
import matplotlib
from tomlkit import key
matplotlib.use("Agg")
import os
import cv2
import logging
from operator import mod
from pip import main
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.metrics import Metrics
from utils.loss_utils import chamfer_sqrt
from SiaTrans.utils import fps_subsample

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

def test_net(cfg, test_data_loader=None, model=None, file_list=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetSubset.TEST),
                                                       batch_size=1,
                                                       num_workers=cfg.CONST.NUM_WORKERS,
                                                       collate_fn=utils.data_loaders.collate_fn,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Switch models to evaluation mode
    model.eval()

    img_list=[]

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            if model_id.split("/")[-1] not in file_list:
                continue

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                img = model.getImg(partial.permute(0, 2, 1).contiguous())
                dir_name = "visualization/asfm"
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                img_croped = crop_img(img)
                cv2.imwrite(dir_name+"/"+taxonomy_id+".png",img_croped)
                img_list.append(img_croped)



if __name__ == "__main__":
    from config_pcn import cfg
    from ASFMNet.SApcn import ASFM
    # taxonomy_map = {
    #     "02691156":"fbc7afa847c30a4c183bb3a05fac486f.pcd",
    #     "02933112":"a0eb46b125a99e26473aef508bd8614e.pcd",
    #     "02958343":"9eaafc3581357b05d52b599fafc842f.pcd",
    #     "03001627":"3de9797fbf88fc35d97fd0ea4791ae93.pcd",
    #     "03636649":"169d73f18f7b17bb4a6ecce2eeb45768.pcd",
    #     "04256520":"f2fbd71be2c50fd89fb1d3c5138b2800.pcd",
    #     "04379243":"ad61a5bc7cba29b88cc413950b617e8f.pcd",
    #     "04530566":"758c75266d7788c0f5678db9e73ab77e.pcd"
    # }

    taxonomy_map = [
        "fbc7afa847c30a4c183bb3a05fac486f.pcd",
        "a0eb46b125a99e26473aef508bd8614e.pcd",
        "9eaafc3581357b05d52b599fafc842f.pcd",
        "3de9797fbf88fc35d97fd0ea4791ae93.pcd",
        "169d73f18f7b17bb4a6ecce2eeb45768.pcd",
        "dd6f44f5b788d5963d6d3c2fb78160fd.pcd",
        "ad61a5bc7cba29b88cc413950b617e8f.pcd",
        "758c75266d7788c0f5678db9e73ab77e.pcd"
    ]

    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                    batch_size=1,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=False)

    model = CandiModel("ASFM", loadParallelModel(ASFM(),"checkpoint/asfm.pth"))
    test_net(cfg,test_data_loader,model=model, file_list=taxonomy_map)