"""
Create uninformative class to extend current available dataset for improving generalization.
The underlinne idea is to mimic an US acquisition that is not yet a lung ultrasound images
To create this 'Uninformative' class the steps are:
- From good acquisition - get the triangular shape of US acquisition
- Take only the zone about the intefrace beetwen probe and tissue
- Random sampling the other zone. 
"""
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F

import time
from tqdm import tqdm
import random
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py

from lung_ultrasound.dataset.dataset_vital import DatasetVitalPOCUS, AugmentationConfig
from lung_ultrasound.tools import cfg_train

def get_reduced_json(original_splitting, percentage, seed=42):
    """
    Create reduced json keeping the same selected subjects across all folds.

    Args:
        original_splitting (dict): original splitting dictionary
        percentage (float): percentage of total unique subjects to keep (e.g., 0.10)
        seed (int): random seed for reproducibility

    Returns:
        dict: reduced splitting dictionary
    """

    random.seed(seed)

    # Collect all unique subjects across all folds
    all_subjects = set()
    for fold in original_splitting.values():
        for split in ["train", "val", "test"]:
            all_subjects.update(fold[split])

    all_subjects = sorted(list(all_subjects))

    # Compute number of subjects to select
    n_selected = max(1, int(len(all_subjects) * percentage))

    selected_subjects = set(random.sample(all_subjects, n_selected))

    # Create reduced dictionary
    reduced_splitting = {}

    for fold_name, fold in original_splitting.items():
        reduced_splitting[fold_name] = {
            "train": [s for s in fold["train"] if s in selected_subjects],
            "val":   [s for s in fold["val"] if s in selected_subjects],
            "test":  [s for s in fold["test"] if s in selected_subjects],
        }

    return reduced_splitting

def get_subject_dict(train_dataset):
    """
    Return subject dict with first 
    """
    print('Creating subject dict...')
    subject_list = set(train_dataset.subject_list)
    
    ## select uninque value of zones
    zones_dict = {sub : []  for sub in set(train_dataset.subject_list)}
    for video, label, subject, zone in train_dataset:
        zones_dict[subject].append(zone)
    for sub in zones_dict.keys():
        zones_dict[sub] = set(zones_dict[sub])

    ## select only the first video
    subjects_dict = {}
    for video, label, subject, zone in train_dataset:  

        if subject not in subjects_dict:
            subjects_dict[subject] = {}

        if zone not in subjects_dict[subject]:
            subjects_dict[subject][zone] = {
                "video": video,
                "label": label
            }

    return subjects_dict

def extract_upper_lower_circular_region(mask,radius_ratio=0.6,center_x_ratio=0.5,center_y_ratio=0.0):
    """
    Split mask into upper (circular arc) and lower complementary region.

    Args:
        mask (torch.Tensor): binary mask (H, W)
        radius_ratio (float): radius as fraction of image height
        center_x_ratio (float): horizontal center as fraction of width
        center_y_ratio (float): vertical center as fraction of height (0 = top)

    Returns:
        upper_mask (torch.Tensor): upper circular region
        lower_mask (torch.Tensor): lower complementary region
    """

    H, W = mask.shape

    cx = int(W * center_x_ratio)
    cy = int(H * center_y_ratio)
    radius = int(H * radius_ratio)

    y = torch.arange(H, device=mask.device).view(-1, 1).expand(H, W)
    x = torch.arange(W, device=mask.device).view(1, -1).expand(H, W)

    circle = ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2

    circular_mask = circle.float()

    upper_mask = mask * circular_mask
    lower_mask = mask * (1 - circular_mask)

    return upper_mask, lower_mask

def create_uninformative_image(img,upper_mask,lower_mask, shape=0.8, scale=0.05):
    """
    Create uninformative image using Gamma distributed noise.

    - Upper region = original image
    - Lower region = Gamma noise

    Args:
        img (torch.Tensor): image (H, W) normalized [0,1]
        upper_mask (torch.Tensor): binary mask (H, W)
        lower_mask (torch.Tensor): binary mask (H, W)
        shape (float): Gamma shape parameter (k)
        scale (float): Gamma scale parameter (θ)

    Returns:
        torch.Tensor: new image (H, W)
    """

    img = img.float()

    # Gamma in PyTorch uses concentration (k) and rate (1/θ)
    gamma_dist = torch.distributions.Gamma(
        concentration=torch.tensor(shape, device=img.device),
        rate=torch.tensor(1.0 / scale, device=img.device)
    )

    noise = gamma_dist.sample(img.shape)

    # Clip per sicurezza (mantieni range realistico)
    noise = torch.clamp(noise, 0.0, 1.0)

    new_img = img * upper_mask + noise * lower_mask

    return new_img

def extract_echo_view(frame, kernel_size=7):
    """
    Extract full white echo view mask from a single 2D frame.

    Args:
        frame (torch.Tensor): 2D tensor (H, W)
        kernel_size (int): size of morphological kernel

    Returns:
        torch.Tensor: binary mask (H, W) with filled echo view
    """

    # Initial binary mask
    mask = (frame > 0).float().unsqueeze(0).unsqueeze(0)

    # Morphological closing (dilation + erosion)
    kernel = torch.ones(
        (1, 1, kernel_size, kernel_size),
        device=frame.device
    )

    # Dilation
    dilated = F.conv2d(mask, kernel, padding=kernel_size // 2)
    dilated = (dilated > 0).float()

    # Erosion
    eroded = F.conv2d(dilated, kernel, padding=kernel_size // 2)
    closed_mask = (eroded == kernel_size * kernel_size).float()

    return closed_mask.squeeze(0).squeeze(0)

def create_uninformative_video(video, radius_ratio=0.3, shape=0.8, scale=0.05):
    """
    Create full uninformative video.

    Args:
        video (torch.Tensor): (C, T, H, W)
    Returns:
        torch.Tensor: uninformative video (C, T, H, W)
    """

    C, T, H, W = video.shape
    new_video = torch.zeros_like(video)

    for t in range(T):

        frame = video[0, t, :, :]

        mask = extract_echo_view(frame)

        upper_mask, lower_mask = extract_upper_lower_circular_region(
            mask,
            radius_ratio=radius_ratio
        )

        new_frame = create_uninformative_image(
            frame,
            upper_mask,
            lower_mask,
            shape=shape,
            scale=scale
        )

        new_video[0, t, :, :] = new_frame

    return new_video


def main(args):
    """
    Create uninformative video clip
    """

    ## read splitting json
    with open(os.path.join(cfg_train.main_path, cfg_train.dataset, cfg_train.splitting), 'r') as file:
        splitting_dict = json.load(file)

    reduced_splitting = get_reduced_json(splitting_dict, 0.10, cfg_train.seed)
    reduced_splitting_path = os.path.join(cfg_train.main_path,cfg_train.dataset, "splitting_10_percent.json")
    with open(reduced_splitting_path, 'w') as f:
        json.dump(reduced_splitting, f, indent=4)

    # configuration without data augmentation
    aug_config = AugmentationConfig(h_flip_p = 0,
                                        v_flip_p = 0.0,
                                        rotation_deg = 0.0,
                                        crop_scale = (1.0, 1.0),
                                        crop_ratio = (1.0, 1.0),
                                        brightness_p = 0.0,
                                        contrast_p = 0.0,
                                        brightness = 0.0,
                                        contrast = 0.0)

    train_dataset = DatasetVitalPOCUS(dataset_path = os.path.join(cfg_train.main_path, cfg_train.dataset),
                                      size = (128,128),
                                      im_channels = cfg_train.im_channels,
                                      lenght = cfg_train.lenght,
                                      overlap = cfg_train.overlap,
                                      fps = cfg_train.fps,
                                      sampling_f = cfg_train.sampling_f,
                                      splitting_json = "splitting_10_percent.json",
                                      fold_cv = cfg_train.fold_cv,
                                      split = 'test', 
                                      normalize = False,
                                      data_augmentation = True,
                                      aug_config = aug_config)
                                      
    print(f'   fold_cv: {cfg_train.fold_cv}')
    print(f'   train dataset: {len(train_dataset)}')
    print('-'*25)

    ## create unformative patient folder
    print('Create uninformative patient folders')
    print(set(train_dataset.subject_list))
    
    for sub in set(train_dataset.subject_list):
        uninformative_path = os.path.join(cfg_train.main_path, cfg_train.dataset, f'{sub}_uninformative')
        os.makedirs(uninformative_path, exist_ok=True)

        for z in train_dataset.zones:
            os.makedirs(os.path.join(uninformative_path,z), exist_ok=True)

    ## get subject dict with first clip of each zones
    subject_dict = get_subject_dict(train_dataset)

    for sub in subject_dict:
        print(sub)
        for zone in subject_dict[sub].keys():
            print(zone)

            video = subject_dict[sub][zone]['video']

            uninformative_video = create_uninformative_video(
                video,
                radius_ratio=0.3,
                shape=0.8,
                scale=0.05
            )

            images_array = np.array(uninformative_video, dtype=np.uint8)
            h5_path = os.path.join(cfg_train.main_path, cfg_train.dataset, f'{sub}_uninformative', zone, f"{zone}.h5")

            with h5py.File(h5_path, "w") as hf:
                hf.create_dataset(
                    "images",
                    data=images_array,
                    compression="gzip",
                    compression_opts=4
                )

            # plt.figure(figsize=(12,4))

            # plt.subplot(1,3,1)
            # plt.imshow(video[0,0], cmap='gray')
            # plt.title("Original First")

            # plt.subplot(1,3,2)
            # plt.imshow(uninformative_video[0,0], cmap='gray')
            # plt.title("Uninformative First")

            # plt.subplot(1,3,3)
            # plt.imshow(uninformative_video[0,-1], cmap='gray')
            # plt.title("Uninformative Last")

            # plt.show()
            
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-classification model for LUS clip')
    parser.add_argument('--keep_log', action='store_true', help='keep the loss,lr, performance during training or not, default=False')

    args = parser.parse_args()

    main(args)