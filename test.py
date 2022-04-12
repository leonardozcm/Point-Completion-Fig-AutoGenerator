# -*- coding: utf-8 -*-
# @Author: XP
# type: ignore
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

def test_net(cfg, test_data_loader=None, model=None):
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

    n_samples = len(test_data_loader)
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()

    # Testing loop
    with tqdm(test_data_loader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)

                partial = data['partial_cloud']
                gt = data['gtcloud']

                b, n, _ = partial.shape

                pcds_pred = model(partial.contiguous())
                cd3 = chamfer_sqrt(pcds_pred, gt).item() * 1e3

                _metrics = [cd3]

                test_metrics.update(_metrics)
                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_map[taxonomy_id], end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

if __name__ == "__main__":
    from config_pcn import cfg
    from SiaTrans.model import SiaTrans
    taxonomy_map = {
        "02691156":"airplane",
        "02933112":"cabinet",
        "02958343":"car",
        "03001627":"chair",
        "03636649":"lamp",
        "04256520":"sofa",
        "04379243":"table",
        "04530566":"watercraft"
    }

    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                    batch_size=1,
                                                    num_workers=cfg.CONST.NUM_WORKERS,
                                                    collate_fn=utils.data_loaders.collate_fn,
                                                    pin_memory=True,
                                                    shuffle=False)

    model = loadParallelModel(SiaTrans(up_factors=[4,8]),"checkpoint/wopa.pth")
    test_net(cfg,test_data_loader,model=model)