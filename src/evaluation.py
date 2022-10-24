'''evaluate output of test'''
import argparse
import numpy as np
from PIL import Image
import os
import glob


def calc_dic(n_objects_gt, n_objects_pred):
    return np.abs(n_objects_gt - n_objects_pred)

def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice

def calc_bd(ins_seg_gt, ins_seg_pred):

    gt_object_idxes = list(set(np.unique(ins_seg_gt)).difference([0]))
    pred_object_idxes = list(set(np.unique(ins_seg_pred)).difference([0]))

    best_dices = []
    for gt_idx in gt_object_idxes:
        _gt_seg = (ins_seg_gt == gt_idx).astype('bool')
        dices = []
        for pred_idx in pred_object_idxes:
            _pred_seg = (ins_seg_pred == pred_idx).astype('bool')

            dice = calc_dice(_gt_seg, _pred_seg)
            dices.append(dice)
        best_dice = np.max(dices)
        best_dices.append(best_dice)

    best_dice = np.mean(best_dices)

    return best_dice

def calc_sbd(ins_seg_gt, ins_seg_pred):

    _dice1 = calc_bd(ins_seg_gt, ins_seg_pred)
    _dice2 = calc_bd(ins_seg_pred, ins_seg_gt)
    return min(_dice1, _dice2)


gt_dir = 'D:/CVPPP2017_LSC_training/images/val'
pred_dir = 'C:/Users/iml/Desktop/SpatialEmbedding_V1_DetailBranch/src/inference/check'

leaf_gt_list = glob.glob(os.path.join(gt_dir, '*_label.png'))
leaf_gt_list.sort()

leaf_pred_list = glob.glob(os.path.join(pred_dir, '*-ins_mask.png'))
leaf_pred_list.sort()

dics, fg_dices, plant_sbds, leaf_sbds = [], [], [], []

for i in range(len(leaf_gt_list)):
    # GT instance mask
    leaf_ins_gt = Image.open(leaf_gt_list[i])
    leaf_ins_gt = np.array(leaf_ins_gt.resize((512, 512), resample=Image.BILINEAR))

    # Instance seg pred
    leaf_ins_pred = np.array(Image.open(leaf_pred_list[i]))

    # SBD (evaluate the performance of instance seg)
    leaf_sbd = calc_sbd(leaf_ins_gt, leaf_ins_pred)
    leaf_sbds.append(leaf_sbd)

    # DiC (leaf instance counting)
    n_leaf_gt = len(np.unique(leaf_ins_gt)[1:])
    n_leaf_pred = len(np.unique(leaf_ins_pred)[1:])

    dic = calc_dic(n_leaf_gt, n_leaf_pred)
    dics.append(dic)

leaf_mean_sbd = np.mean(leaf_sbds)
mean_dic = np.mean(dics)

print('LEAF MEAN SBD : ', leaf_mean_sbd)
print ('MEAN |DIC|   : ', mean_dic)