"""
Set training options.
"""
import copy
import os

import torch
from utils import transforms as my_transforms

DATASET_DIR='D:/CVPPP2017_LSC_training'

args = dict(

    cuda=True,
    display=True,
    display_it=5,

    save=True,
    save_dir='D:/SpatialEmbedding_V1_DetailBranch/exp',
    resume_path=None,

    train_dataset = {
        'name': 'cvppp2',  # (cityscapes/cvppp/cvppp2), cvppp2: more augmentation than cvppp
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'train',
            'size': None,
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'instance', 'label'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 4,
        'workers': 0
    },

    val_dataset = {
        'name': 'cvppp2',
        'kwargs': {
            'root_dir': DATASET_DIR,
            'type_': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ['image', 'instance', 'label'],
                        'type': [torch.FloatTensor, torch.ByteTensor, torch.ByteTensor],
                    }
                },
            ]),
        },
        'batch_size': 1,
        'workers': 0
    },

    # if n_sigma=1, num_classes should be [3, 1] and vice versa
    # if n_sigma=2, num_classes should be [4, 1] and vice versa
    # check the test_config_cvppp too
    model = {
        'name': 'branched-bisenetv2-custom_ca1',  # branched-erfnet / branched-erfnet2 / branched-unet / branched-bisenetv2
        'kwargs': {
            'num_classes': [3, 1],  # 3 for offsets(x, y) & sigma / 1 for seed map
        }
    },

    lr=1e-3,
    n_epochs=500, # every x epochs, report train & validation set

    # loss options
    loss_opts={
        # to_center=True -> center=mean of x, y coordinate map / False -> center=mean of spatial embedding
        # when calculating loss
        'to_center': False,
        'n_sigma': 1,  # 1 for circular margin, 2 for elliptical margin
        'foreground_weight': 10  # for instance loss (lovasz-hinge)
    },

    loss_w={
        'w_inst': 1,
        'w_var': 10,



        'w_seed': 1
    },
)

def get_args():
    return copy.deepcopy(args)