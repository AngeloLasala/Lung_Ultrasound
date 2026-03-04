from torch.autograd import Variable
import os
import numpy as np
import torch
import torch.nn.functional as F
from hausdorff import hausdorff_distance
import time
import pandas as pd
import logging
import lung_ultrasound.quality_model.utils.metrics as metrics

def eval_mask_slice(valloader, model, criterion, cfg):
    """
    Evaluate mask slice by slice
    """
    model.eval()
    
    val_losses = 0
    max_slice_number = cfg.batch_size * (len(valloader) + 1)
    dices = np.zeros((max_slice_number, cfg.num_classes))
    hds = np.zeros((max_slice_number, cfg.num_classes))
    ious = np.zeros((max_slice_number, cfg.num_classes))
    accs = np.zeros((max_slice_number, cfg.num_classes))
    ses = np.zeros((max_slice_number, cfg.num_classes))
    sps = np.zeros((max_slice_number, cfg.num_classes))
    
    eval_number = 0
    sum_time = 0

    for batch_idx, datapack in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=cfg.device))
        labels = Variable(datapack['label'].to(device=cfg.device))
            
        with torch.no_grad():
            pred = model(imgs)  # pred: B x n_class x H x W

        # Calcolo loss
        val_loss = criterion(pred, labels)
        val_losses += val_loss.item()

        # Predizione argmax
        predict = torch.argmax(pred, dim=1).detach().cpu().numpy()  # B x H x W
        gt = labels.detach().cpu().numpy()  # B x H x W

        b, h, w = gt.shape
        for j in range(b):
            for c in range(cfg.num_classes):
                pred_mask = (predict[j] == c).astype(np.uint8)
                gt_mask = (gt[j] == c).astype(np.uint8)

                # Calcolo metriche
                dice_i = metrics.dice_coefficient(pred_mask, gt_mask)
                dices[eval_number+j, c] += dice_i

                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_mask, gt_mask, all=False)
                ious[eval_number+j, c] += iou
                accs[eval_number+j, c] += acc
                ses[eval_number+j, c] += se
                sps[eval_number+j, c] += sp

                # Hausdorff distance
                hds[eval_number+j, c] += hausdorff_distance(pred_mask, gt_mask, distance="manhattan")
        
        eval_number += b

    # Rimuovo slice non usate
    dices = dices[:eval_number, :] 
    hds = hds[:eval_number, :] 
    ious = ious[:eval_number, :]
    accs = accs[:eval_number, :]
    ses = ses[:eval_number, :]
    sps = sps[:eval_number, :]

    val_losses /= (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices, axis=0)
    hd_mean = np.mean(hds, axis=0)
    hd_std = np.std(hds, axis=0)

    mean_dice = np.mean(dice_mean[1:])  # media escludendo classe 0 (background)
    mean_hdis = np.mean(hd_mean[1:])

    if cfg.mode == "train":
        return dices, mean_dice, mean_hdis, val_losses
    else:
        dice_mean *= 100
        dices_std *= 100
        iou_mean = np.mean(ious*100, axis=0)
        iou_std = np.std(ious*100, axis=0)
        acc_mean = np.mean(accs*100, axis=0)
        acc_std = np.std(accs*100, axis=0)
        se_mean = np.mean(ses*100, axis=0)
        se_std = np.std(ses*100, axis=0)
        sp_mean = np.mean(sps*100, axis=0)
        sp_std = np.std(sps*100, axis=0)
        return dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean, dices_std, hd_std, iou_std, acc_std, se_std, sp_std