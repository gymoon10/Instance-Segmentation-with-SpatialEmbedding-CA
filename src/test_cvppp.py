"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import time

import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm

import test_config_cvppp  # for CVPPP
import torch
from datasets import get_dataset
from models import get_model
from utils.utils_cvppp import Cluster, Visualizer  # for CVPPP
import numpy as np
from PIL import Image

torch.backends.cudnn.benchmark = True

args = test_config_cvppp.get_args()

if args['display']:
    plt.ion()
else:
    plt.ioff()
    plt.switch_backend("agg")

if args['save']:
    if not os.path.exists(args['save_dir']):
        os.makedirs(args['save_dir'])

# set device
device = torch.device("cuda:0" if args['cuda'] else "cpu")

# dataloader
dataset = get_dataset(
    args['dataset']['name'], args['dataset']['kwargs'])
dataset_it = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0, pin_memory=True if args['cuda'] else False)

# load model
model = get_model(args['model']['name'], args['model']['kwargs'])
model = torch.nn.DataParallel(model).to(device)

state = torch.load(args['checkpoint_path'])
model.load_state_dict(state['model_state_dict'], strict=True)
model.eval()

# cluster module
num_classes = args['model']['kwargs']['num_classes']
n_sigma = 0
if sum(num_classes) == 4:
    n_sigma = 1
else:
    n_sigma = 2

cluster = Cluster()

# Visualizer
visualizer = Visualizer(('image', 'pred', 'sigma', 'seed'))

with torch.no_grad():
    for sample in tqdm(dataset_it):

        im = sample['image']
        instances = sample['instance'].squeeze()

        output = model(im)
        instance_map, predictions, mask = cluster.cluster(output[0], n_sigma=n_sigma, threshold=0.9)  # ths of center

        if args['display']:
            visualizer.display(im, 'image')

            visualizer.display([instance_map.cpu(), instances.cpu()], 'pred')

            sigma = output[0][2].cpu()
            sigma = (sigma - sigma.min()) / (sigma.max() - sigma.min())
            sigma[instances == 0] = 0
            visualizer.display(sigma, 'sigma')

            seed = torch.sigmoid(output[0][3]).cpu()
            visualizer.display(seed, 'seed')

        if args['save']:

            base, _ = os.path.splitext(os.path.basename(sample['im_name'][0]))

            txt_file = os.path.join(args['save_dir'], base + '.txt')
            with open(txt_file, 'w') as f:

                # loop over instances
                for id, pred in enumerate(predictions):
                    im_name = base + '_{:02d}.png'.format(id)
                    im = torchvision.transforms.ToPILImage()(
                        pred['mask'].unsqueeze(0))

                    # write image
                    im.save(os.path.join(args['save_dir'], im_name))

                    # write to file
                    cl = 26
                    score = pred['score']
                    f.writelines("{} {} {:.02f}\n".format(im_name, cl, score))

            # save semantic-mask : discarded
            #sem_seg_pred = (mask * 1).cpu().detach().numpy()[0]
            #sem_seg_pred = sem_seg_pred.astype(np.uint8)
            #sem_seg_pred = sem_seg_pred * 255
            #sem_seg_pred_pil = Image.fromarray(sem_seg_pred)

            name = sample['im_name'][0].split('\\')[-1][:-4]
            #print(os.path.join(args['save_dir'], name + '-fg_mask.png'))
            #sem_seg_pred_pil.save(os.path.join(args['save_dir1'], name + '-fg_mask.png'))

            # save instance-mask
            _n_clusters = len(np.unique(instance_map.flatten())) - 1  # discard bg
            colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
            ins_seg_pred_color = np.zeros(
                (instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
            for i in range(_n_clusters):
                ins_seg_pred_color[instance_map == (
                        i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

            # color
            ins_seg_pred_color_pil = Image.fromarray(ins_seg_pred_color)
            ins_seg_pred_color_pil.save(os.path.join(args['save_dir1'], name + '-ins_mask_color.png'))

            # gray
            ins_seg_pred_pil = Image.fromarray(instance_map.numpy())
            ins_seg_pred_pil.save(os.path.join(args['save_dir1'], name + '-ins_mask.png'))


