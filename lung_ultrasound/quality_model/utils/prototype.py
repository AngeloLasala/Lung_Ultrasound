"""
Advanced evaluation of pleura and rib's shadow annotation to get more information
about the geometries of Gold Standard acquisitions and not gold standard
"""
import os
from random import randint
import numpy as np
import torch
from skimage import color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
import os
import cv2
import pandas as pd
import random
import tqdm
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from lung_ultrasound.quality_model.utils.visualization import visualize_image_mask
from lung_ultrasound.quality_model.cfg import cfg
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.evaluation import eval_mask


def compute_symmetry_ribs(regions):
    """
    Symmetry index based on pairwise differences of relative region sizes.

    Returns:
        0.0   → sentinel: class absent (n=0), no geometric meaning
        1.0   → single region (n=1) or perfect symmetry (n≥2)
        [0,1] → n≥2, 0=max asymmetry, 1=perfect symmetry
    """
    n = len(regions)

    if n == 0:
        return 0.0  # sentinel: class absent

    if n == 1:
        return 1.0  # natural limit of the formula

    sizes = np.array([r['relative_size_to_class'] for r in regions])

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(abs(sizes[i] - sizes[j]))

    mean_diff = np.mean(pairs)
    return float(1.0 - mean_diff)


def compute_pleura_parameters(regions, mask_width, mask_height, n_max=5):
    """
    Compute geometrical parameters for pleura regions.

    Args:
        regions     : list of dicts with keys:
                        'relative_size_to_class' : float
                        'centroid_x'             : float
                        'centroid_y'             : float
        mask_width  : int, W
        mask_height : int, H
        n_max       : int, maximum expected number of regions (hyperparameter)

    Returns:
        dict with:
            p1 : lateral position in x,          [-1, 1], 0=centered
            p2 : vertical position in y,          [-1, 1], 0=at H/3
            p3 : compactness (inverted entropy),  [0,  1], 1=single region
            p4 : fragment asymmetry,              [-1, 1], 0=no dominant fragment
            gold standard: p* = [0, 0, 1, 0]
    """
    n = len(regions)

    if n == 0:
        return {
            'p1': float('nan'),
            'p2': float('nan'),
            'p3': float('nan'),
            'p4': float('nan'),
            'centroid_x': float('nan'),
            'centroid_y': float('nan'),
        }

    sizes = np.array([r['relative_size_to_class'] for r in regions])

    # ── Global centroid (weighted by region size) ────────────────────────────
    centroid_x = float(np.sum(sizes * np.array([r['centroid_x'] for r in regions])))
    centroid_y = float(np.sum(sizes * np.array([r['centroid_y'] for r in regions])))

    # ── p1: lateral position in x, [-1, 1] ──────────────────────────────────
    p1 = (centroid_x / mask_width - 0.5) * 2.0

    # ── p2: vertical position in y, [-1, 1] ─────────────────────────────────
    y_norm = centroid_y / mask_height - 1/3
    if centroid_y < mask_height / 3:
        p2 = y_norm / (1/3)   # normalize by distance to top:    H/3
    else:
        p2 = y_norm / (2/3)   # normalize by distance to bottom: 2H/3

    # ── p3: compactness (inverted entropy) ───────────────────────────────────
    if n == 1:
        p3 = 1.0
    else:
        safe_sizes = sizes[sizes > 0]
        entropy    = -np.sum(safe_sizes * np.log(safe_sizes))
        p3         = 1.0 - entropy / np.log(n_max)

    # # ── p4: spatial skewness of fragments, [-1, 1] ───────────────────────────
    if n == 1:
        p4 = 0.0
    else:
        x_cents = np.array([r['centroid_x'] for r in regions])
        x_norm  = (x_cents - centroid_x) / (mask_width / 2.0)  # normalized positions

        variance = float(np.sum(sizes * x_norm**2))
        if variance < 1e-10:
            p4 = 0.0  # all fragments at same position
        else:
            skewness = float(np.sum(sizes * x_norm**3)) / (variance ** 1.5)
            # clip to [-1, 1] — skewness is unbounded in theory
            p4 = float(np.tanh(skewness))

    # ── p4: fragment asymmetry, [-1, 1]  (covariance) ─────────────────────────────────────
    # if n == 1:
    #     p4 = 0.0  # single region, no asymmetry
    # else:
    #     s_mean  = np.mean(sizes)
    #     x_cents = np.array([r['centroid_x'] for r in regions])
    #     x_norm  = (x_cents - centroid_x) / (mask_width / 2.0)
    #     p4      = float(np.sum((sizes - s_mean) * x_norm))

    return {
        'p1': float(p1),
        'p2': float(p2),
        'p3': float(p3),
        'p4': float(p4),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
    }


def compute_ribs_parameters(regions, mask_width, n_max=6):
    """
    Compute geometrical parameters for ribs regions.

    Returns:
        r1 : lateral position in x,  [-1, 1], 0=centered
        r2 : spatial skewness,       [-1, 1], 0=symmetric
        r3 : entropy compactness,    [0,  1], 1=exactly 2 uniform regions
        gold standard: r* = [0, 0, 1]
    """
    n = len(regions)

    if n == 0:
        return {
            'r1': float('nan'), 'r2': float('nan'), 'r3': float('nan'),
            'centroid_x': float('nan'), 'centroid_y': float('nan'),
        }

    sizes = np.array([r['relative_size_to_class'] for r in regions])

    # ── Global centroid (weighted by region size) ────────────────────────────
    centroid_x = float(np.sum(sizes * np.array([r['centroid_x'] for r in regions])))
    centroid_y = float(np.sum(sizes * np.array([r['centroid_y'] for r in regions])))

    # ── r1: lateral position in x, [-1, 1] ──────────────────────────────────
    r1 = (centroid_x / mask_width - 0.5) * 2.0

    # ── r2: spatial skewness, [-1, 1] ────────────────────────────────────────
    if n == 1:
        r2 = 0.0
    else:
        x_cents = np.array([r['centroid_x'] for r in regions])
        x_norm  = (x_cents - centroid_x) / (mask_width / 2.0)
        variance = float(np.sum(sizes * x_norm**2))
        if variance < 1e-10:
            r2 = 0.0
        else:
            skewness = float(np.sum(sizes * x_norm**3)) / (variance ** 1.5)
            r2 = float(np.tanh(skewness))

    # ── r3: entropy distance from log(2), [0, 1] ────────────────────────────
    if n == 1:
        r3 = 1.0 - np.log(2) / np.log(n_max)
    else:
        safe_sizes = sizes[sizes > 0]
        entropy    = -np.sum(safe_sizes * np.log(safe_sizes))
        r3         = 1.0 - abs(entropy - np.log(2)) / np.log(n_max)

    return {
        'r1': float(r1),
        'r2': float(r2),
        'r3': float(r3),
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
    }


def get_parameters_from_mask(mask, n_max_pleura=5, n_max_ribs=6):
    """
    Get geometrical parameters from masks.

    Args:
        mask          : numpy array (H, W) with pixel values:
                        0 = background
                        1 = pleura
                        2 = ribs
        n_max_pleura  : max expected pleura regions (entropy normalization)
        n_max_ribs    : max expected ribs regions   (entropy normalization)

    Returns:
        dict with parameters for pleura and ribs
    """
    total_pixels = mask.size
    mask_height, mask_width = mask.shape[:2]

    # ── helper ──────────────────────────────────────────────────────────────
    def extract_regions(binary_mask, class_total_pixels):
        num_regions, labels = cv2.connectedComponents(binary_mask)
        regions = []
        for region_id in range(1, num_regions):  # 0 is background
            region_mask        = (labels == region_id)
            region_pixel_count = int(region_mask.sum())
            row_indices        = np.where(region_mask)[0]  # y
            col_indices        = np.where(region_mask)[1]  # x
            regions.append({
                'region_id'             : region_id,
                'pixel_count'           : region_pixel_count,
                'relative_size'         : region_pixel_count / total_pixels,
                'relative_size_to_class': region_pixel_count / class_total_pixels if class_total_pixels > 0 else 0.0,
                'centroid_x'            : float(col_indices.mean()),
                'centroid_y'            : float(row_indices.mean()),
            })
        return regions

    # ── Pleura (label = 1) ──────────────────────────────────────────────────
    pleura_binary       = (mask == 1).astype(np.uint8)
    pleura_total_pixels = int(pleura_binary.sum())
    pleura_regions      = extract_regions(pleura_binary, pleura_total_pixels)
    pleura_symmetry     = compute_symmetry_ribs(pleura_regions)
    p_params            = compute_pleura_parameters(pleura_regions, mask_width, mask_height, n_max=n_max_pleura)

    # ── Ribs (label = 2) ────────────────────────────────────────────────────
    ribs_binary         = (mask == 2).astype(np.uint8)
    ribs_total_pixels   = int(ribs_binary.sum())
    ribs_regions        = extract_regions(ribs_binary, ribs_total_pixels)
    ribs_symmetry       = compute_symmetry_ribs(ribs_regions)
    r_params            = compute_ribs_parameters(ribs_regions, mask_width, n_max=n_max_ribs)

    return {
        'pleura': {
            'total_pixels'  : pleura_total_pixels,
            'relative_size' : pleura_total_pixels / total_pixels,
            'num_regions'   : len(pleura_regions),
            'regions'       : pleura_regions,
            'symmetry_index': pleura_symmetry,
            'p1'            : p_params['p1'],         # centrality in x, [0,1]
            'p2'            : p_params['p2'],         # vertical position, [0,1]
            'p3'            : p_params['p3'],         # compactness, [0,1]
            'p4'            : p_params['p4'],
            'centroid_x'    : p_params['centroid_x'],
            'centroid_y'    : p_params['centroid_y'],
        },
        'ribs': {
            'total_pixels'  : ribs_total_pixels,
            'relative_size' : ribs_total_pixels / total_pixels,
            'num_regions'   : len(ribs_regions),
            'regions'       : ribs_regions,
            'symmetry_index': ribs_symmetry,
            'r1'            : r_params['r1'],         # centrality in x, [0,1]
            'r2'            : r_params['r2'],         # shape symmetry, [0,1]
            'r3'            : r_params['r3'],         # entropy target, [0,1]
            'centroid_x'    : r_params['centroid_x'],
            'centroid_y'    : r_params['centroid_y'],
        },
        'total_pixels' : total_pixels,
        'mask_width'   : mask_width,
        'mask_height'  : mask_height,
    }


def main(args):
    """
    Compute info about the geometries of gold standard and not gold standard acquisitions
    """

    transform = JointTransform2D(img_size=cfg.img_size, low_img_size=cfg.img_size,
                                 ori_size=cfg.img_size,
                                 crop=cfg.crop,
                                 p_flip=0,
                                 p_rota=0,
                                 p_scale=0,
                                 p_gaussn=0,
                                 p_contr=0,
                                 p_gama=0,
                                 p_distor=0,
                                 color_jitter_params=cfg.color_jitter_params,
                                 long_mask=cfg.long_mask)
    dataset_list = []
    for splitt in ['train', 'val']:
        dataset_i = LungDataset(dataset_path=os.path.join(cfg.main_path, cfg.dataset),
                                img_size=cfg.img_size,
                                fold_cv=cfg.fold_cv,
                                splitting_json=cfg.splitting,
                                split=splitt,
                                joint_transform=transform,
                                one_hot_mask=False)
        dataset_list.append(dataset_i)
    dataset = torch.utils.data.ConcatDataset(dataset_list)

    # ── Accumulators ────────────────────────────────────────────────────────
    pleura_params_log = {'p1': [], 'p2': [], 'p3': [], 'p4': []}
    ribs_params_log   = {'r1': [], 'r2': [], 'r3': []}

    for data in tqdm.tqdm(dataset):
        mask   = data['label'].detach().cpu().numpy()
        params = get_parameters_from_mask(mask)

        pleura_params_log['p1'].append(params['pleura']['p1'])
        pleura_params_log['p2'].append(params['pleura']['p2'])
        pleura_params_log['p3'].append(params['pleura']['p3'])
        pleura_params_log['p4'].append(params['pleura']['p4'])

        ribs_params_log['r1'].append(params['ribs']['r1'])
        ribs_params_log['r2'].append(params['ribs']['r2'])
        ribs_params_log['r3'].append(params['ribs']['r3'])

        # ── Visualization ────────────────────────────────────────────────────
        # fig, axes = plt.subplots(1, 2, figsize=(10, 5), num=f"{data['subject']}_{data['zone']}")

        # axes[0].imshow(data['image'].permute(1,2,0).numpy(), cmap='gray')
        # axes[0].set_title("Img")

        # axes[1].imshow(data['image'].permute(1,2,0).numpy(), cmap='gray')
        # axes[1].imshow(data['label'], alpha=0.2, cmap='jet')

        # # gold standard targets
        # gs_x = params['mask_width']  / 2.0
        # gs_y = params['mask_height'] / 3.0
        # axes[1].scatter(gs_x, gs_y,
        #                 c='yellow', s=200, marker='*',
        #                 zorder=5, label='GS pleura target')

        # # pleura centroid
        # if not np.isnan(params['pleura']['centroid_x']):
        #     axes[1].scatter(params['pleura']['centroid_x'], params['pleura']['centroid_y'],
        #                     c='cyan', s=80, marker='o', zorder=5,
        #                     label=f"Pleura centroid "
        #                           f"(p1={params['pleura']['p1']:.2f}, "
        #                           f"p2={params['pleura']['p2']:.2f}, "
        #                           f"p3={params['pleura']['p3']:.2f}, "
        #                           f"p4={params['pleura']['p4']:.2f}, "
        #                           )

        # # ribs centroid
        # if not np.isnan(params['ribs']['centroid_x']):
        #     axes[1].scatter(params['ribs']['centroid_x'], params['ribs']['centroid_y'], 
        #                     c='orange', s=80, marker='s', zorder=5,
        #                     label=f"Ribs centroid "
        #                           f"(r1={params['ribs']['r1']:.2f}, "
        #                           f"r2={params['ribs']['r2']:.2f}, "
        #                           f"r3={params['ribs']['r3']:.2f})")

        # axes[1].legend(loc='lower right', fontsize=7)
        # axes[1].set_title("Img + mask + centroids")

        # plt.tight_layout()
        # plt.show()

    # ── Plot parameters ──────────────────────────────────────────────────────
    tau_pleura = {'p1': 0.15, 'p2': 0.15, 'p3': 0.15, 'p4': 0.15}
    tau_ribs   = {'r1': 0.15, 'r2': 0.15, 'r3': 0.15}

    configs = [
        {
            'params_log': pleura_params_log,
            'gs'        : {'p1': 0.0, 'p2': 0.0, 'p3': 1.0, 'p4': 0.0},
            'tau'       : tau_pleura,
            'pairs'     : [('p1', 'p2'), ('p3', 'p4'), ('p1', 'p3')],
            'color'     : 'tomato',
            'title'     : 'Pleura',
        },
        {
            'params_log': ribs_params_log,
            'gs'        : {'r1': 0.0, 'r2': 0.0, 'r3': 1.0},
            'tau'       : tau_ribs,
            'pairs'     : [('r1', 'r2'), ('r1', 'r3'), ('r2', 'r3')],
            'color'     : 'steelblue',
            'title'     : 'Ribs',
        },
    ]

    for cfg_plot in configs:
        n_pairs = len(cfg_plot['pairs'])
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))

        for ax, (px, py) in zip(axes, cfg_plot['pairs']):
            ax.scatter(cfg_plot['params_log'][px], cfg_plot['params_log'][py],
                       c=cfg_plot['color'], s=10, alpha=0.2, zorder=3, label='Samples')

            gx, gy   = cfg_plot['gs'][px], cfg_plot['gs'][py]
            tx, ty   = cfg_plot['tau'][px], cfg_plot['tau'][py]

            ax.scatter(gx, gy, c='gold', s=200, marker='*', zorder=5, label='Gold standard', alpha=0.6)

            rect = patches.Rectangle(
                (gx - tx, gy - ty), 2 * tx, 2 * ty,
                color='gold', alpha=0.4, zorder=2, label=f'GS region (±{tx}, ±{ty})'
            )
            ax.add_patch(rect)

            # set axis limits based on parameter range
            xlim = (-1.05, 1.05) if px in ('p1', 'p2', 'p4', 'r1', 'r2') else (-0.05, 1.05)
            ylim = (-1.05, 1.05) if py in ('p1', 'p2', 'p4', 'r1', 'r2') else (-0.05, 1.05)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ax.set_xlabel(px, fontsize=12)
            ax.set_ylabel(py, fontsize=12)
            ax.set_title(f'{px} vs {py}', fontsize=13)
            ax.set_aspect('equal')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"{cfg_plot['title']} parameters — distribution vs gold standard", fontsize=14)
        plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset for semantic segmentation')
    args = parser.parse_args()
    main(args)