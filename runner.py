from tkinter.messagebox import NO
import torch
from utils.loss_utils import chamfer_sqrt
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from visualization.test_vis import get_ptcloud_img
from utils.io import IO
import numpy as np


class CandiModel(object):
    def __init__(self, name, path, model) -> None:
        self.name = name
        self.path = path
        self.model = torch.nn.DataParallel(model).cuda()

        checkpoint = torch.load(self.path)
        self.model.load_state_dict(checkpoint['model'])
    
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


def test_net(test_data_loader=None, model=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True
    model.eval()
    with tqdm(test_data_loader) as t:
        for _, (_, file_path, data) in enumerate(t):
            if not result.has_key(file_path):
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
        for _, (_, file_path, data) in enumerate(t):
            if file_path not in files:
                continue

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']
                # b, n, 3

                cd = model.ge (partial.contiguous(), gt)

                result[file_path][model.name]=cd


def select_outperforms():
    file_name = []
    for file, pair in result.items():
        file_name.append((file, pair["SnowflakeNet"]-pair["Ours"]))
    
    sorted(file_name, key=lambda x:x[1], reverse=True)
    
    return file_name[:10]


       

result = {}
dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
    utils.data_loaders.DatasetSubset.TEST),
                                                batch_size=1,
                                                num_workers=cfg.CONST.NUM_WORKERS,
                                                collate_fn=utils.data_loaders.collate_fn,
                                                pin_memory=True,
                                                shuffle=False)

Model_list = [
    CandiModel("GRNet", "checkpoint/grnet.pth", ),
    CandiModel("PCN", "checkpoint/pcn.pth", ),
    CandiModel("PMPNet", "checkpoint/pmpnet.pth", ),
    CandiModel("SnowflakeNet", "checkpoint/snowflakenet.pth", ),
    CandiModel("Ours", "checkpoint/ours.pth", ),
]

for model in Model_list[-2:]:
    test_net(test_data_loader, model)

file_select = select_outperforms()

transforms = utils.data_transforms.Compose([{
                'callback': 'UpSamplePoints',
                'parameters': {
                    'n_points': cfg.DATASETS.SHAPENET.N_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
