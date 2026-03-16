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
from lung_ultrasound.quality_model.models.losses import CombinedCEDiceLoss
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.evaluation import eval_mask
# from preoperativeSAM.utils.evaluation import get_eval


def main(args):
    """
    Train Unet for Semantic segmentation
    """

    ## set logging level  ###########################################################
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    ## set train configuration and logging configuration ###########################################################
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate
    device = torch.device(cfg.device)

    ## Overwrite cfg.fold from parser
    if args.fold is not None:
        cfg.fold_cv = args.fold
    
    if args.splitting is not None:
        cfg.splitting = args.splitting
    
    # main file for results
    if args.keep_log:
        logtimestr = args.timestamp if args.timestamp is not None else time.strftime('%d-%m-%Y_%H-%M')
        logging.info(f' Day: {logtimestr}\n')

        results_path = os.path.join(cfg.main_path, cfg.results, cfg.dataset, cfg.model_name)
        if not os.path.isdir(results_path):
            os.makedirs(results_path)

        checkpoint_path = os.path.join(results_path, cfg.fold_cv, logtimestr, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        boardpath = os.path.join(results_path, cfg.fold_cv, logtimestr, 'tensorboard') #, args.modelname, opt.tensorboard_folder, f'{args.dataset_loader}_{logtimestr}')
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    ## set random seed for reproducibility  #######################################
    seed_value = cfg.seed           
    np.random.seed(seed_value)                        # set random seed for numpy
    random.seed(seed_value)                           # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)    # avoid hash random
    torch.manual_seed(seed_value)                     # set random seed for CPU
    torch.cuda.manual_seed(seed_value)                # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)            # set random seed for all GPU
    torch.backends.cudnn.deterministic = True 
    
    ## get the model  ##############################################################
    logging.info(f" Creating model: {cfg.model_name} ...")
    if cfg.model_name == 'UNet':
        model  = UNet(in_channels=cfg.im_channels, num_classes=cfg.num_classes, base_filters=64, bilinear=True).to(device)
    n  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f' N parameters: {n/1e6:.2f}M')
    logging.info(' Done!\n')
    logging.info(f'  num_classes: {cfg.num_classes} (included background)')
    logging.info('-'*25)
    
    ## load the dataset  ##########################################################
    logging.info(' Creating train and val dataloader...')
    tf_train = JointTransform2D(img_size = cfg.img_size, 
                                low_img_size = cfg.low_img_size, 
                                ori_size = cfg.img_size, 
                                crop = cfg.crop, 
                                p_flip = cfg.p_flip, 
                                p_rota = cfg.p_rota, 
                                p_scale = cfg.p_scale, 
                                p_gaussn = cfg.p_gaussn,
                                p_contr = cfg.p_contr, 
                                p_gama = cfg.p_gama, 
                                p_distor = cfg.p_distor, 
                                color_jitter_params = cfg.color_jitter_params, 
                                long_mask = cfg.long_mask)  # image prerocessing and aug

    tf_val = JointTransform2D(img_size = cfg.img_size, 
                                low_img_size = cfg.low_img_size, 
                                ori_size = cfg.img_size, 
                                crop = cfg.crop, 
                                p_flip = 0.0, 
                                p_rota = 0.0, 
                                p_scale = 0.0, 
                                p_gaussn = 0.0,
                                p_contr = 0.0, 
                                p_gama = 0.0, 
                                p_distor = 0.0, 
                                color_jitter_params = None, 
                                long_mask = cfg.long_mask)  # image prerocessing

    train_dataset = LungDataset(dataset_path = os.path.join(cfg.main_path, cfg.dataset),
                                img_size = cfg.img_size,
                                fold_cv = cfg.fold_cv,
                                splitting_json = cfg.splitting,
                                split = 'train', 
                                joint_transform = tf_train, 
                                one_hot_mask = False)

    val_dataset = LungDataset(dataset_path = os.path.join(cfg.main_path, cfg.dataset),
                            img_size = cfg.img_size,
                            fold_cv = cfg.fold_cv,
                            splitting_json = cfg.splitting,
                            split = 'val', 
                            joint_transform = tf_val, 
                            one_hot_mask = False) # return image, mask, and filename
    
    trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    logging.info(f'  splitting: {cfg.splitting}')
    logging.info(f'  - train dataset: {len(train_dataset)}')
    logging.info(f'  - val dataset: {len(val_dataset)}')
    logging.info('-'*25)

    ## Train initialization ########################################################################
    logging.info(' Train initialization...')
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    learning_rate = cfg.learning_rate

    logging.info(f'  device: {device}')
    logging.info(f'  epochs: {epochs}')
    logging.info(f'  batch_size: {batch_size}')
    logging.info(f'  learning_rate: {learning_rate}')
    logging.info(f'  Loss = {cfg.w_ce} * L_ce + {cfg.w_dice} * L_dice')
    logging.info(f'  size: {cfg.size}')
    logging.info(f'  device: {device}')
    
    ## optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
   
    ## loss function
    class_weights = torch.tensor(cfg.class_weights, device=device)
    logging.info(f'  class weights: {class_weights}')
    criterion = CombinedCEDiceLoss(weight=class_weights, w_ce=cfg.w_ce, w_dice=cfg.w_dice)

    logging.info(' Done!\n')
 
    ## Model training ################################################################################################
    iter_num = 0
    max_iterations = epochs * len(trainloader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(epochs+1), np.zeros(epochs+1)

    logging.info('Start training ...')        
    for epoch in range(epochs):
        progress_bar = tqdm(total=len(trainloader), disable=False)
        progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(trainloader):
            imgs = datapack['image'].to(dtype = torch.float32, device=device)
            masks = datapack['label'].to(device=device)
            
            ## forward
            pred = model(imgs)
            train_loss = criterion(pred, masks)

            ## backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": train_loss.detach().item()})

        if args.keep_log:
            # TensorWriter.add_scalar('loss/train', train_losses / (batch_idx + 1), epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        ## evaluation
        if epoch % cfg.eval_freq == 0:
            model.eval()

            dices, mean_dice, _, val_losses, dice_per_class = eval_mask(valloader, model, criterion=criterion, cfg=cfg, device=device)
            progress_bar.set_postfix({"loss": train_losses / (batch_idx + 1), 
                                      'val loss': val_losses,
                                      'val dice': mean_dice,
                                      })

            if args.keep_log:
                ## logger scalar
                TensorWriter.add_scalars('loss', {'train': train_losses / (batch_idx + 1), 'val': val_losses}, epoch)
                TensorWriter.add_scalars('dice', {'mean': mean_dice, 'Pleura': dice_per_class[1], 'Ribs': dice_per_class[2]}, epoch)                      
                dice_log[epoch] = mean_dice

                # logger images
                rand_batch = random.choice(list(valloader))
                imgs_val = rand_batch['image'].to(dtype=torch.float32, device=device)
                masks_val = rand_batch['label'].to(device=device)

                rand_idx = random.randint(0, imgs_val.shape[0] - 1)
                img_sample   = imgs_val[rand_idx]    # (C, H, W)
                mask_sample  = masks_val[rand_idx]   # (C, H, W) or (H, W)
                
                with torch.no_grad():
                    pred_logits = model(imgs_val[rand_idx].unsqueeze(0))  # (1, num_classes, H, W)
                pred_sample = torch.argmax(pred_logits, dim=1).squeeze(0)  # (H, W) class indices

                TensorWriter.add_image('val/image', img_sample[:1], epoch)
                TensorWriter.add_image('val/label', mask_sample.unsqueeze(0) * 100, epoch)
                TensorWriter.add_image('val/predicted', pred_sample.unsqueeze(0) * 100, epoch)


                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'best_model.pth'))
                    # logging.info(f' saving best model at epoch:{epoch}\n')
            
    ## save last model
    if args.keep_log:
        save_path = os.path.join(checkpoint_path, f'last_model.pth')
        torch.save(model.state_dict(), save_path)
        logging.info(f'  --> saved last model at epoch {epoch+1}')

        cfg_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("__")}
        with open(os.path.join(results_path, cfg.fold_cv, logtimestr, 'train_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=4)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet model for semantic segmentation')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--fold', type=str, default=None, help='Cross-validation fold (e.g. fold_1)')
    parser.add_argument('--splitting', type=str, default=None, help='Splitting JSON file (e.g. splitting_v2.json)')
    parser.add_argument('--timestamp', type=str, default=None, help='Shared timestamp for all folds (e.g. 16-03-2026_10-00)')
    parser.add_argument('--keep_log', action='store_true', help='keep the loss,lr, performance during training or not, default=False')

    args = parser.parse_args()

    main(args)