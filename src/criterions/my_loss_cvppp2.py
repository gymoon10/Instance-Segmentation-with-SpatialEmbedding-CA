"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

Add cosine inter loss (neighborings)
"""
import math

import numpy as np

import torch
import torch.nn as nn
from criterions.lovasz_losses import lovasz_hinge

import os
import numpy as np
import cv2
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt


def get_neighbor_by_distance(label_map, distance=10, max_neighbor=32):
    label_map = label_map.copy()

    def _adjust_size(x):
        if len(x) >= max_neighbor:
            return x[0:max_neighbor]
        else:
            return np.pad(x, (0, max_neighbor - len(x)), 'constant', constant_values=(0, 0))

    # idx of instance labels
    unique = np.unique(label_map)
    assert unique[0] == 0
    if len(unique) <= 2:  # only one instance
        return None

    neighbor_indice = np.zeros((max_neighbor, max_neighbor))
    label_flat = label_map.reshape((-1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (distance * 2 + 1, distance * 2 + 1))

    for i, label in enumerate(unique[1:]):
        assert i + 1 == label
        mask = (label_map == label)  # mask of specific instance
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).reshape((-1))

        neighbor_pixel_ind = np.logical_and(dilated_mask > 0, label_flat != 0)
        neighbor_pixel_ind = np.logical_and(neighbor_pixel_ind, label_flat != label)

        neighbors = np.unique(label_flat[neighbor_pixel_ind])
        neighbor_indice[i + 1, :] = _adjust_size(neighbors)  # padding

    return neighbor_indice.astype(np.int32)


class SpatialEmbLoss2(nn.Module):

    def __init__(self, to_center=True, n_sigma=1, foreground_weight=1, ):
        super().__init__()

        print('Created Spatial Embedding Loss2 with: to_center: {}, n_sigma: {}, foreground_weight: {}'.format(
            to_center, n_sigma, foreground_weight))

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # pixel coordinate map for 512x512
        xm = torch.linspace(0, 1, 512).view(
            1, 1, -1).expand(1, 512, 512)  # x-coordinates are within [0, 1] : (1, h, w)
        ym = torch.linspace(0, 1, 512).view(
            1, -1, 1).expand(1, 512, 512)  # y-coordinates are within [0, 1] : (1, h, w)
        xym = torch.cat((xm, ym), 0)

        self.register_buffer("xym", xym)

    def forward(self, prediction, instances, labels, w_inst=1, w_var=10, w_seed=1, iou=False, iou_meter=None):
        '''prediction : model output (N, 4, h, w)
        instances : GT instance-mask (N, h, w)
        labels : GT semantic-mask (N, h, w)'''

        batch_size, height, width = prediction.size(
            0), prediction.size(2), prediction.size(3)

        xym_s = self.xym[:, 0:height, 0:width].contiguous()  # (2, h, w)

        loss = 0

        for b in range(0, batch_size):

            ### Instance-branch ###
            # pixel-embeddings = offset vectors + coordinate vectors
            spatial_emb = torch.tanh(prediction[b, 0:2]) + xym_s  # (2, h, w)
            # sigma value for each pixels (large simga -> bigger margin)
            sigma = prediction[b, 2:2 + self.n_sigma]  # (n_sigma, h, w)

            ### Seed-branch ###
            # seed map for each semantic classes (object-ness score map, [0, 1])
            # pixel-embeddings who lay close to their instance center -> high seed score
            seed_map = torch.sigmoid(
                prediction[b, 2 + self.n_sigma:2 + self.n_sigma + 1])  # (1, h, w)

            # loss accumulators
            var_loss = 0
            instance_loss = 0  # lovasz-hinge
            seed_loss = 0
            obj_count = 0
            # added losses to original spatial embedding loss
            inter_cosine_loss = 0
            intra_loss = 0

            # GTs
            instance = instances[b].unsqueeze(0)  # (1, h, w)
            label = labels[b].unsqueeze(0)  # (1, h, w)

            # instance lists except bg
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids != 0]

            # seed loss 1 - bg pixels are regressed to 0
            bg_mask = (label == 0)
            if bg_mask.sum() > 0:
                seed_loss += torch.sum(
                    torch.pow(seed_map[bg_mask] - 0, 2))

            # for each instance
            for id in instance_ids:

                # mask of specific instance k (RoI pixels of instance k)
                # spatial_emb, sigma, seed_map below only consider the pixels of RoI k
                in_mask = instance.eq(id)  # (1, h, w)

                # calculate center of attraction (default: to_center=False)
                # if to_center=True -> mean of x, y coordinate maps which belong to instance pixels
                if self.to_center:
                    xy_in = xym_s[in_mask.expand_as(xym_s)].view(2, -1)
                    center = xy_in.mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # defining center as the mean over the pixel-embeddings of instance k
                else:
                    center = spatial_emb[in_mask.expand_as(spatial_emb)].view(
                        2, -1).mean(1).view(2, 1, 1)  # 2 x 1 x 1

                # calculate sigma belonging to specific instance
                sigma_in = sigma[in_mask.expand_as(
                    sigma)].view(self.n_sigma, -1)

                # sigma_k = avg of all sigma_i belongs to instance k
                s = sigma_in.mean(1).view(
                    self.n_sigma, 1, 1)  # n_sigma x 1 x 1

                # calculate var loss before exp - e.q (5)
                var_loss = var_loss + \
                           torch.mean(
                               torch.pow(sigma_in - s[..., 0].detach(), 2))

                s = torch.exp(s * 10)  # https://github.com/davyneven/SpatialEmbeddings/issues/4

                # calculate gaussian output - e.q (9)
                # it outputs a fg/bg probability map (distance from current instance center)
                # high value near the current instance center, low value at the boundary area
                dist = torch.exp(-1 * torch.sum(
                    torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True))  # (1, h, w), 1 for current instance k

                # apply lovasz-hinge loss
                # it optimizes the i.o.u between gaussian output & GT mask of current instance
                instance_loss = instance_loss + \
                                lovasz_hinge(dist * 2 - 1, (in_mask * 1).cuda())  # *1 to convert boolean to int

                # seed loss - e.q (10)
                # fg pixels of seed map are regressed to the gaussian output of pixel embedding
                # seed_map[in_mask]: torch.size([N])
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2))


                # ------ Intra Loss --------------
                in_label = in_mask[0].cpu().detach().numpy().flatten()
                shape = spatial_emb.permute(1, 2, 0).size()  # (h, w, 2)

                in_locations = torch.LongTensor(np.where(in_label == True)[0]).type(torch.LongTensor)
                in_locations = in_locations.cuda()
                in_embeddings = torch.index_select(
                    spatial_emb.permute(1, 2, 0).contiguous().view(shape[0] * shape[1], shape[2]),
                    dim=0, index=in_locations).type(torch.DoubleTensor)
                in_embeddings = in_embeddings.type(torch.float32)

                # mean of gaussian dist from spatial embeddings and mean embedding of current instance
                gaussian_dist2mean = torch.exp(
                    -1 * torch.sum(torch.pow(in_embeddings.cuda() - center.view(2, ), 2) * s, 2)).mean()
                intra_loss += (1 - gaussian_dist2mean)


                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1


            # -----Inter Loss-----
            # 1. calculate mean embedding of each instance area
            emb_means = []
            embedding_flat = spatial_emb.view(2, -1)
            max_label = int(instance.unique().max().item())

            for label in range(0, max_label + 1):  # include bg
                mask = (instance == label).flatten()  # roi pixels (pixels belonging to current instance)
                count = mask.sum()

                # embeddings belong to current instance area
                mask_embedding = embedding_flat[:, mask]  # (E_channels, count)
                mean = torch.sum(mask_embedding, dim=1) / count  # mean embedding  (2, )
                emb_means.append(mean)
            emb_means = torch.stack(emb_means, dim=0)

            distance = 5
            max_neighbor = 32

            # get ids of neighboring instances
            # neighbor[4] - ids of neighboring instances of 4th instance
            neighbor = get_neighbor_by_distance(instance[0].cpu().detach().numpy(), distance, max_neighbor)

            bg_include_n_labels = len(emb_means)  # bg + all instances
            cos_sim = torch.nn.CosineSimilarity(dim=1)

            main_means = emb_means.unsqueeze(1).expand(bg_include_n_labels, bg_include_n_labels, 2)
            neighbor_means = main_means.clone().permute(1, 0, 2)
            main_means = main_means.reshape(-1, 2)  # (bg_include_n_labels^2, 2)
            neighbor_means = neighbor_means.reshape(-1, 2)  # (bg_include_n_labels^2, 2)

            inter = cos_sim(neighbor_means, main_means).view(bg_include_n_labels, bg_include_n_labels).abs()

            inter_mask = torch.zeros(bg_include_n_labels, bg_include_n_labels, dtype=torch.float)
            for main_label in range(1, bg_include_n_labels):
                for nei_label in neighbor[main_label]:  # neighbor[i-1]: ids of neighboring instances of i th instance
                    if nei_label == 0:
                        break
                    inter_mask[main_label][nei_label] = 1.0

            inter_mask = inter_mask.cuda()
            inter_mean = torch.sum(inter * inter_mask) / torch.sum(inter_mask)
            inter_cosine_loss += inter_mean


            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count
                inter_cosine_loss /= obj_count

            seed_loss = seed_loss / (height * width)

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss + 5*intra_loss

        loss = loss / (b + 1)
        #print('loss :', loss)
        #print('instance_loss :', w_inst * instance_loss)
        #print('var_loss :', w_var * var_loss)
        #print('seed_loss :', w_seed * seed_loss)
        #print()

        return loss + prediction.sum() * 0


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
