"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import os

from PIL import Image

import torch
from utils import transforms as my_transforms

CVPPP_DIR = 'D:/CVPPP2017_LSC_training'

args = dict(

    cuda=True,
    display=True,

    save=True,
    save_dir='C:/Users/iml/Desktop/SpatialEmbedding_V1_DetailBranch/src/masks/',
    save_dir1='C:/Users/iml/Desktop/SpatialEmbedding_V1_DetailBranch/src/inference/',  # additional
    checkpoint_path='D:/SpatialEmbedding_V1_DetailBranch/exp/BiSeNetV2_Custom_CA1_bias=False/best_iou_model.pth',

    dataset={
        'name': 'cvppp2',
        'kwargs': {
            'root_dir': CVPPP_DIR,
            'type_': 'val',
            'transform': my_transforms.get_transform([
                {
                    'name': 'ToTensor',
                    'opts': {
                        'keys': ('image', 'instance', 'label'),
                        'type': (torch.FloatTensor, torch.ByteTensor, torch.ByteTensor),
                    }
                },
            ]),
        }
    },

    model={
        # check models/_init_.py
        'name': "branched-bisenetv2-custom_ca1",
        'kwargs': {
            'num_classes': [3, 1],
        }
    }
)


def get_args():
    return copy.deepcopy(args)
