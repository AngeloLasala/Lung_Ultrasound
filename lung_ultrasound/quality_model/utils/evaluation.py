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
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def eval_mask(valloader, model, criterion, cfg):
    """
    Evaluate model on single images (no slice aggregation).

    Args:
        valloader : DataLoader yielding {'image': Tensor, 'label': Tensor}
        model     : segmentation model (output: B x num_classes x H x W)
        criterion : CombinedLoss or any loss that returns (loss, dict) | scalar
        cfg       : config object with fields:
                        .device, .num_classes, .batch_size, .mode ('train'|'test')
    Returns:
        train mode : dices, mean_dice, mean_hd, val_loss, dice_mean
        test  mode : dice_mean, hd_mean, iou_mean, acc_mean, se_mean, sp_mean,
                     dices_std, hd_std, iou_std, acc_std, se_std, sp_std
    """
    # Allow cfg to be passed as a plain dict
    if isinstance(cfg, dict):
        from types import SimpleNamespace
        cfg = SimpleNamespace(**cfg)
    print('ok')
    model.eval()

    n_samples = cfg.batch_size * len(valloader)   # upper bound
    dices = np.zeros((n_samples, cfg.num_classes))
    hds = np.zeros((n_samples, cfg.num_classes))
    ious = np.zeros((n_samples, cfg.num_classes))
    accs = np.zeros((n_samples, cfg.num_classes))
    ses = np.zeros((n_samples, cfg.num_classes))
    sps = np.zeros((n_samples, cfg.num_classes))

    val_losses = 0.0
    sample_idx = 0          # global sample counter (replaces inner j)

    for batch_idx, datapack in enumerate(valloader):
        imgs   = Variable(datapack['image'].to(dtype=torch.float32, device=cfg.device))
        labels = Variable(datapack['label'].to(device=cfg.device))

        with torch.no_grad():
            pred = model(imgs)           # B x num_classes x H x W

        # ── Loss ──────────────────────────────────────────────────────────
        # CombinedLoss returns (tensor, dict); plain losses return a tensor
        loss_out = criterion(pred, labels)
        loss_val = loss_out[0] if isinstance(loss_out, tuple) else loss_out
        val_losses += loss_val.item()

        # ── Predictions ───────────────────────────────────────────────────
        predict = torch.argmax(pred, dim=1).detach().cpu().numpy()  # B x H x W
        gt = labels.detach().cpu().numpy()                     # B x H x W
        
        B = gt.shape[0]
        for j in range(B):
            idx = sample_idx + j          # ← correct global index
            for c in range(cfg.num_classes):
                pred_mask = (predict[j] == c).astype(np.uint8)
                gt_mask   = (gt[j]      == c).astype(np.uint8)

                pred_mask = np.expand_dims(pred_mask, axis=0)
                gt_mask = np.expand_dims(gt_mask, axis=0)
                
                dices[idx, c] = metrics.dice_coefficient(pred_mask, gt_mask)

                iou, acc, se, sp = metrics.sespiou_coefficient2(pred_mask, gt_mask, all=False)
                ious[idx, c] = iou
                accs[idx, c] = acc
                ses[idx,  c] = se
                sps[idx,  c] = sp

                hds[idx, c] = hausdorff_distance(pred_mask[0], gt_mask[0], distance="manhattan")

                # import matplotlib.pyplot as plt
                # plt.figure()
                # plt.title(f'pred {c}')
                # plt.imshow(pred_mask[0])

                # plt.figure()
                # plt.title(f'label {c}')
                # plt.imshow(gt_mask[0])
                # plt.show()

        sample_idx += B

    # ── Trim to actual number of evaluated samples ─────────────────────────
    dices = dices[:sample_idx]
    hds   = hds[:sample_idx]
    ious  = ious[:sample_idx]
    accs  = accs[:sample_idx]
    ses   = ses[:sample_idx]
    sps   = sps[:sample_idx]

    val_losses /= (batch_idx + 1)

    dice_mean = np.mean(dices, axis=0)
    dices_std = np.std(dices,  axis=0)
    hd_mean   = np.mean(hds,   axis=0)
    hd_std    = np.std(hds,    axis=0)
    # Exclude background (class 0) from summary scalars
    mean_dice = np.mean(dice_mean[1:])
    mean_hd   = np.mean(hd_mean[1:])
    
    # ── Return ─────────────────────────────────────────────────────────────
    if cfg.mode == "train":
        return dices, mean_dice, mean_hd, val_losses, dice_mean

    # test mode: scale to percentage
    dice_mean  *= 100;  dices_std *= 100
    iou_mean    = np.mean(ious * 100, axis=0)
    iou_std     = np.std(ious  * 100, axis=0)
    acc_mean    = np.mean(accs * 100, axis=0)
    acc_std     = np.std(accs  * 100, axis=0)
    se_mean     = np.mean(ses  * 100, axis=0)
    se_std      = np.std(ses   * 100, axis=0)
    sp_mean     = np.mean(sps  * 100, axis=0)
    sp_std      = np.std(sps   * 100, axis=0)

    return (
        dice_mean, hd_mean,  iou_mean,  acc_mean,  se_mean,  sp_mean,
        dices_std, hd_std,   iou_std,   acc_std,   se_std,   sp_std,
    )