import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy.core.fromnumeric import transpose
from PIL import Image
from torch.utils.data import Dataset


class CVPPPDataset(Dataset):

    def __init__(self, root_dir='./', type_="train", size=None, transform=None):
        self.root_dir = root_dir
        self.type = type_

        # get image, foreground and instance list
        image_list = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_rgb.png'))
        image_list.sort()
        self.image_list = image_list
        print("# image files: ", len(image_list))

        if self.type != 'test':
            instance_list = glob.glob(os.path.join(self.root_dir, 'images/{}/'.format(self.type), '*_label.png'))
            instance_list.sort()
            self.instance_list = instance_list
            print("# label files: ", len(instance_list))

        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform

        self.jitter = transforms.ColorJitter(brightness=0.03,
                                             contrast=0.03,
                                             saturation=0.03,
                                             hue=0.03)

        print('CVPPP Dataset created [{}]'.format(self.type))

    def __len__(self):

        return self.real_size if self.size is None else self.size

    def __getitem__(self, index):
        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image and foreground
        image = Image.open(self.image_list[index]).convert('RGB')
        image = image.resize((512, 512), resample=Image.BILINEAR)
        # black_canvas = Image.new("RGB", image.size, 0)
        # fg = fg.resize((512,512), resample=Image.NEAREST).convert('L')
        # image = Image.composite(image, black_canvas, fg) # remove background

        width, height = image.size
        sample['image'] = image
        sample['im_name'] = self.image_list[index]

        if self.type != 'test':
            # convert labels to instance map
            instance_map = skimage.io.imread(self.instance_list[index])  # := instance map
            instance_map = cv2.resize(instance_map, (512, 512), interpolation=cv2.INTER_NEAREST)
            instance_map = cv2.cvtColor(instance_map, cv2.COLOR_RGBA2GRAY)
            instance_ids = np.unique(instance_map)[1:]  # no background

            instance = np.zeros((height, width), dtype=np.uint8)
            label = np.zeros((height, width), dtype=np.uint8)
            instance_counter = 0
            for instance_id in instance_ids:
                instance_counter = instance_counter + 1
                mask = (instance_map == instance_id)

                instance[mask] = instance_counter
                label[mask] = 1

        # --- data augmentation ---
        if self.type == 'train':
            # random hflip
            if random.random() > 0.5:
                # FLIP_TOP_BOTTOM
                sample['image'] = sample['image'].transpose(Image.FLIP_LEFT_RIGHT)
                instance = np.flip(instance, axis=0)
                label = np.flip(label, axis=0)

            # random vflip
            if random.random() > 0.5:
                # FLIP_LEFT_RIGHT
                sample['image'] = sample['image'].transpose(Image.FLIP_TOP_BOTTOM)
                instance = np.flip(instance, axis=1)
                label = np.flip(label, axis=1)

            # random jittering
            if random.random() > 0.5:
                # need to applied on PIL Image
                sample['image'] = self.jitter(sample['image'])

        if self.type != 'test':
            label = Image.fromarray(np.uint8(label))
            instance = Image.fromarray(np.uint8(instance))

            sample['instance'] = instance
            sample['label'] = label

        # transform
        if (self.transform is not None):
            sample = self.transform(sample)

        return sample