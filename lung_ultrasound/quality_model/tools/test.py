"""
Train model for LUS semantic segmentation
"""

import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torchvision

import time
from tqdm import tqdm
import random
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
import json
import numpy as np

from lung_ultrasound.quality_model.cfg import cfg
from lung_ultrasound.quality_model.models.unet import UNet
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.evaluation import eval_mask

def main(args):
    """
    Train Unet for Semantic segmentation
    """

    ## set logging level  ###########################################################
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    ## Read trained files
    checkpoints_path = os.path.join(args.model_path, 'checkpoints')
    cfg_train_path = os.path.join(args.model_path, 'train_config.json')

    ## read cfg_train configuration file
    if not os.path.exists(cfg_train_path):
        raise FileNotFoundError(f"File not found: {cfg_train_path}")

    with open(cfg_train_path, 'r') as f:
        cfg = json.load(f)

    device = torch.device(cfg['device'])
    
    ## create test dataset
    logging.info(' Creating test dataloader...')
    
    tf_test = JointTransform2D(img_size = cfg['img_size'], 
                                low_img_size = cfg['low_img_size'], 
                                ori_size = cfg['img_size'], 
                                crop = cfg['crop'], 
                                p_flip = 0.0, 
                                p_rota = 0.0, 
                                p_scale = 0.0, 
                                p_gaussn = 0.0,
                                p_contr = 0.0, 
                                p_gama = 0.0, 
                                p_distor = 0.0, 
                                color_jitter_params = None, 
                                long_mask = cfg['long_mask'])  # image prerocessing

    test_dataset = LungDataset(dataset_path = os.path.join(cfg['main_path'], cfg['dataset']),
                                img_size = cfg['img_size'],
                                fold_cv = cfg['fold_cv'],
                                splitting_json = cfg['splitting'],
                                split = 'test', 
                                joint_transform = tf_test, 
                                one_hot_mask = False)
    
    testloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'], pin_memory=True)
    logging.info(f"   fold_cv: {cfg['fold_cv']}")
    logging.info(f'  - test dataset: {len(test_dataset)}')
    logging.info('-'*25)

    ## Train initialization ########################################################################
    logging.info(' Creating model...')
    if cfg['model_name'] == 'UNet':
        model  = UNet(in_channels=cfg['im_channels'], num_classes=cfg['num_classes'], base_filters=64, bilinear=True).to(device)
    best_checkpoint = os.path.join(checkpoints_path, f'{args.saved_model}_model.pth')

    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")
    logging.info(f" Loading checkpoint from {best_checkpoint}")

    checkpoint = torch.load(best_checkpoint, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode

    logging.info(f"  num_classes: {cfg['num_classes']}")
    logging.info('-'*25)
    
    ## loss function
    class_weights = torch.tensor(cfg['class_weights'], device=device)
    logging.info(f'  class weights: {class_weights}')
    criterion = nn.CrossEntropyLoss(weight=class_weights)
 
    ## Model training 
    model.eval()
    dices, mean_dice, _, val_losses, dice_per_class = eval_mask(testloader, model, criterion=criterion, cfg=cfg)
    print(f'mean dice (no background) = {mean_dice:.4f}')
    print(f'Pleura dice = {dice_per_class[1]:.4f}')
    print(f'Ribs shadow dice = {dice_per_class[2]:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet model for semantic segmentation')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--saved_model', type=str, help='Select best or last')
    args = parser.parse_args()

    main(args)