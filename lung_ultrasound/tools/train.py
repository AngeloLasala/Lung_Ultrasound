"""
Main file to train the LUS multi instance classification. The code provides flags for hyperparameters tuning and directory path where load the data 
as well as save the output.
See cfg_train in lung_ultrasound/tools/__init__.py for configuration details.
"""
import os
import argparse

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader

import time
from tqdm import tqdm
import random
import logging
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.ERROR)


from lung_ultrasound.dataset.dataset_LUS_covid import DatasetLUSCovid
from lung_ultrasound.models.resnet import ResNetLUSCAM
from lung_ultrasound.models.vgg16 import VGG16LUSCAM
from lung_ultrasound.losses.bce import WeightedBCELoss, compute_pos_weights
from lung_ultrasound.tools import cfg_train

def main(args):
    """
    Main function for training the LUS multi-instance classifier.
    """

    ## set logging level  ###########################################################
    logging_dict = {'debug':logging.DEBUG, 'info':logging.INFO, 'warning':logging.WARNING, 'error':logging.ERROR, 'critical':logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    ## set train configuration and logging configuration ###########################################################
    epochs = cfg_train.epochs
    batch_size = cfg_train.batch_size
    learning_rate = cfg_train.learning_rate
    device = torch.device(cfg_train.device)

    # main file for results
    if args.keep_log:
        logtimestr = time.strftime('%d-%m-%Y_%H-%M')  # initialize the tensorboard for record the training process
        logging.info(f' Day: {logtimestr}\n')

        results_path = os.path.join(cfg_train.main_path, cfg_train.results, cfg_train.dataset, cfg_train.backbone)
        if not os.path.isdir(results_path):
            os.makedirs(results_path)

        checkpoint_path = os.path.join(results_path, cfg_train.fold_cv, logtimestr, 'checkpoints')
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)

        boardpath = os.path.join(results_path, cfg_train.fold_cv, logtimestr, 'tensorboard') #, args.modelname, opt.tensorboard_folder, f'{args.dataset_loader}_{logtimestr}')
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)

    ## set random seed for reproducibility  #######################################
    seed_value = cfg_train.seed           
    np.random.seed(seed_value)                        # set random seed for numpy
    random.seed(seed_value)                           # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)    # avoid hash random
    torch.manual_seed(seed_value)                     # set random seed for CPU
    torch.cuda.manual_seed(seed_value)                # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)            # set random seed for all GPU
    torch.backends.cudnn.deterministic = True         # set random seed for convolution

    ## create model ###########################################################
    logging.info(' Creating model...')
    model = ResNetLUSCAM(num_classes=cfg_train.num_classes,
                         backbone=cfg_train.backbone,
                         pretrained=cfg_train.pretrained,
                         freeze_backbone=cfg_train.freeze_backbone,
                         pooling=cfg_train.pooling)

    # model = VGG16LUSCAM(num_classes=cfg_train.num_classes,
    #                      backbone=cfg_train.backbone,
    #                      pretrained=cfg_train.pretrained,
    #                      freeze_backbone=cfg_train.freeze_backbone,
    #                      pooling=cfg_train.pooling)


    ## to do: add part to compute model complexity and numeber of parameters
    logging.info(f'  num_classes: {cfg_train.num_classes}')
    logging.info(f'  backbone: {cfg_train.backbone}')
    logging.info(f'  pretrained: {cfg_train.pretrained}')
    logging.info(f'  freeze_backbone: {cfg_train.freeze_backbone}')
    logging.info(f'  pooling: {cfg_train.pooling}')
    logging.info('-'*25)

    ## load dataset ###########################################################
    logging.info(' Creating train and val dataloader...')
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                                         transforms.Resize(cfg_train.size),  
                                        ])

    inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    train_dataset = DatasetLUSCovid(dataset_path = os.path.join(cfg_train.main_path, cfg_train.dataset),
                                    data_augmentation = True,
                                    size = cfg_train.size,
                                    im_channels = cfg_train.im_channels,
                                    splitting_json='splitting.json',
                                    fold_cv = cfg_train.fold_cv,
                                    split = 'train',
                                    trasformations=transformation)

    val_dataset = DatasetLUSCovid(dataset_path = os.path.join(cfg_train.main_path, cfg_train.dataset),
                                    data_augmentation = False,
                                    size = cfg_train.size,
                                    im_channels = cfg_train.im_channels,
                                    splitting_json='splitting.json',
                                    fold_cv = cfg_train.fold_cv,
                                    split = 'val', 
                                    trasformations=transformation)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    logging.info(f'   fold_cv: {cfg_train.fold_cv}')
    logging.info(f'   train dataset: {len(train_dataset)}')
    logging.info(f'   val dataset: {len(val_dataset)}')
    logging.info('-'*25)

    ## Train initialization ########################################################################    
    logging.info(' Training initialization...')
    logging.info(f'  epochs: {epochs}')
    logging.info(f'  batch_size: {batch_size}')
    logging.info(f'  learning_rate: {learning_rate}')
    logging.info(f'  size: {cfg_train.size}')
    logging.info(f'  device: {device}')

    ## lr scheduler
    if cfg_train.cosine_annealing:
        logging.info(f'  using cosine annealing lr from {learning_rate} to {learning_rate/10}')
        base_lr = learning_rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate/10)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    logging.info('-'*25)

    #initial best loss
    best_loss = float('inf')
    loss_log = np.zeros(epochs+1)

    # criterion
    pos_weight = compute_pos_weights(torch.stack([label for _, label, _ in train_dataset], dim=0))
    criterion = WeightedBCELoss(pos_weight=pos_weight)

    model.to(device)
    for epoch in range(epochs):
        progress_bar = tqdm(total=len(trainloader), disable=False)
        progress_bar.set_description(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        train_losses = 0
        for batch_idx, (images, labels, subj) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device).float() 
            
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses += loss.item()


            progress_bar.set_postfix({'loss': train_losses/(batch_idx+1), 'lr': optimizer.param_groups[0]['lr']})
            progress_bar.update(1)
        
        ## adjust learning rate
        if cfg_train.cosine_annealing:
            scheduler.step()

        if args.keep_log:
            TensorWriter.add_scalar('Train/Loss', train_losses/len(trainloader), epoch)
            TensorWriter.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        # Validation
        if epoch % cfg_train.eval_freq == 0:
            model.eval()
            val_losses = 0
            for batch_idx, (images, labels, subj) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device).float()

                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_losses += loss.item()
            avg_val_loss = val_losses / len(valloader)

            if args.keep_log:
                TensorWriter.add_scalar('Val/Loss', avg_val_loss, epoch)

                ## to do: aggiugnere parte di visualizzaione image e cam
                idx_v = np.random.randint(0, len(val_dataset))
                image_v, label_v, subj_v = val_dataset[idx_v]
                image_v_input = image_v.unsqueeze(0).to(device)
                label_v_input = label_v.unsqueeze(0).to(device).float()

                model.eval()
                with torch.no_grad():
                    output_v = model(image_v_input)
                    outpunt_v_sigmoid = torch.sigmoid(output_v).cpu().numpy()[0]
                #     cam_v = model.get_cam_weights()
                #     print(cam_v.shape, output_v)
                #     exit()
                
                ## create figure
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                image_v = inv_normalize(image_v)  # denormalize for visualization
                image_v = image_v.permute(1, 2, 0).cpu().numpy()  # convert to HWC format for visualization
                axes[0].imshow(image_v, cmap='gray')
                axes[0].set_title(f"Label {label_v_input}")

                axes[1].imshow(image_v, cmap='gray')
                # axes[1].set_title(f'Output {outpunt_v_sigmoid}')

                TensorWriter.add_figure('Val/Image_and_Output', fig, epoch)
                plt.close(fig)

            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                # save model
                save_path = os.path.join(checkpoint_path, f'best_model.pth')
                torch.save(model.state_dict(), save_path)
                # logging.info(f'  --> saved best model with val loss: {best_loss:.4f} at epoch {epoch+1}')

            ## save last model
            if epoch == epochs - 1:
                save_path = os.path.join(checkpoint_path, f'last_model.pth')
                torch.save(model.state_dict(), save_path)
                logging.info(f'  --> saved last model at epoch {epoch+1}')

    cfg_dict = {k: v for k, v in cfg_train.__dict__.items() if not k.startswith("__")}
    with open(os.path.join(results_path, cfg_train.fold_cv, logtimestr, 'train_config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LUS Multi-Instance Classifier')
    # parser.add_argument('--data_dir', type=str, required=True,
    #                     help='Path to the dataset directory')
    parser.add_argument('--log', type=str, default='debug', help='Logging level')
    parser.add_argument('--keep_log', action='store_true', help='keep the loss,lr, performance during training or not, default=False')

    args = parser.parse_args()

    main(args)

