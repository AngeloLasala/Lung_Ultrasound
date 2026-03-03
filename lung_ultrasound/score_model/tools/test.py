"""
Main test file for evaluting the performance on test set.
This is a version compatible with the Pisani's model trained on recent available dataset

Note: the predicted class are slightly different from the paper of 'Clinical benefict ...'
"""
import os
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import time
from tqdm import tqdm
import random
import logging
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.ERROR)

from lung_ultrasound.score_model.dataset.dataset_vital import DatasetVitalPOCUS, AugmentationConfig
from lung_ultrasound.score_model.losses.cce import WeightedCrossEntropyLoss, compute_class_weights
from lung_ultrasound.score_model.tools import cfg_train, load_model
from lung_ultrasound.score_model.tools.helper import EMA, EarlyStopping
import lung_ultrasound.utils as utils


def main(args):
    """
    Statistical summary on test set for a treined model
    """
    classes_name = ["Normal", "B-lines", "Consolidation", "B-lines + Consolidations", "Indeterminate"]

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
        cfg_train = json.load(f)

    device = torch.device(cfg_train['device'])
    
    ## create test dataset
    logging.info(' Creating test dataloader...')
   
    aug_config_test = AugmentationConfig(h_flip_p = 0,
                                        v_flip_p = 0.0,
                                        rotation_deg = 0.0,
                                        crop_scale = (1.0, 1.0),
                                        crop_ratio = (1.0, 1.0),
                                        brightness_p = 0.0,
                                        contrast_p = 0.0,
                                        brightness = 0.0,
                                        contrast = 0.0)

    train_dataset = DatasetVitalPOCUS(dataset_path = os.path.join(cfg_train['main_path'], cfg_train['dataset']),
                                      size = cfg_train['size'],
                                      im_channels = cfg_train['im_channels'],
                                      lenght = cfg_train['lenght'],
                                      overlap = cfg_train['overlap'],
                                      fps = cfg_train['fps'],
                                      sampling_f = cfg_train['sampling_f'],
                                      splitting_json = cfg_train['splitting'],
                                      fold_cv = cfg_train['fold_cv'],
                                      split = 'train', 
                                      normalize = True,
                                      data_augmentation = True,
                                      aug_config = aug_config_test)

    logging.info(f" Train statistic: mean={train_dataset.mean:.4f} - std={train_dataset.std:.4f}")
                                      
    test_dataset = DatasetVitalPOCUS(dataset_path = os.path.join(cfg_train['main_path'], cfg_train['dataset']),
                                      size = cfg_train['size'],
                                      im_channels = cfg_train['im_channels'],
                                      lenght = cfg_train['lenght'],
                                      overlap = cfg_train['overlap'],
                                      fps = cfg_train['fps'],
                                      sampling_f = cfg_train['sampling_f'],
                                      splitting_json = cfg_train['splitting'],
                                      fold_cv = cfg_train['fold_cv'],
                                      split = 'test', 
                                      normalize = True,
                                      data_augmentation = True,
                                      aug_config = aug_config_test,
                                      mean = train_dataset.mean,      ## This is important: YOU HAVE TO NORMALIZE WITH MEAN AND STD OF TRAINING DATASET
                                      std = train_dataset.std)

    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    logging.info(f"   fold_cv: {cfg_train['fold_cv']}")
    logging.info(f'   test dataset: {len(test_dataset)}')
    logging.info('-'*25)

    ## create model ###########################################################
    logging.info(' Creating model...')
    model = load_model(cfg_train)
    best_checkpoint = os.path.join(checkpoints_path, f'{args.saved_model}_ema_model.pth')

    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")
    logging.info(f" Loading checkpoint from {best_checkpoint}")

    checkpoint = torch.load(best_checkpoint, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    logging.info(f"  num_classes: {cfg_train['num_classes']}")
    logging.info('-'*25)

    ## Performe the inferece
    all_labels, all_preds = [], []
    for batch_idx, (videos, labels, subject, zones) in enumerate(testloader):
        videos, labels, subject, zones = videos.to(device), labels, subject, zones
        subject = str(subject[0]) 
        zones = str(zones[0])
        labels = labels.numpy()

        with torch.no_grad():
            outputs = model(videos)[0].to(device)
        out_index = torch.argmax(outputs, dim=1).detach().cpu().numpy()

        print(f" {subject} ({zones}): label:{classes_name[labels[0]]} - prediction:{classes_name[out_index[0]]}") 
        all_labels.append(labels[0])
        all_preds.append(out_index[0])   

    report, fig, stratified_chance, majority_chance = utils.confusion_matrix(all_labels, all_preds, classes_name)

    ## create test_results folder
    results_folder = os.path.join(args.model_path, 'test_results')
    os.makedirs(results_folder, exist_ok=True)

    report_path = os.path.join(results_folder, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    chance_dict = {
        "stratified_random_accuracy": float(stratified_chance),
        "majority_class_accuracy": float(majority_chance)
    }
    chance_path = os.path.join(results_folder, 'chance_levels.json')
    with open(chance_path, 'w') as f:
        json.dump(chance_dict, f, indent=4)

    fig_path = os.path.join(results_folder, 'confusion_matrix.png')
    fig.savefig(fig_path, dpi=300)
    logging.info(f" Saved confusion matrix figure to: {fig_path}")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-classification model for LUS clip')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--saved_model', type=str, help='Select best or last')


    args = parser.parse_args()

    main(args)

