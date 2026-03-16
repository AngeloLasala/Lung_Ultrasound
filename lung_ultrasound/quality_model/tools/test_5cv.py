"""
Test model for LUS semantic segmentation - 5-fold cross validation
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn
import logging
logging.getLogger("numba").setLevel(logging.WARNING)
import json
import numpy as np
import matplotlib.pyplot as plt

from lung_ultrasound.quality_model.models.unet import UNet
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.evaluation import eval_mask



def test_fold(fold, model_path, timestamp, data_path, saved_model, device):
    """
    Run test on a single fold and return metrics + per-image results
    """
    fold_path = os.path.join(model_path, fold, timestamp)
    checkpoints_path = os.path.join(fold_path, 'checkpoints')
    cfg_train_path = os.path.join(fold_path, 'train_config.json')

    ## load train config
    if not os.path.exists(cfg_train_path):
        raise FileNotFoundError(f"Config not found: {cfg_train_path}")
    with open(cfg_train_path, 'r') as f:
        cfg = json.load(f)

    ## override main_path if provided
    if data_path is not None:
        logging.info(f"  Overriding main_path: {cfg['main_path']} -> {data_path}")
        cfg['main_path'] = data_path

    ## create test dataset
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
                                long_mask = cfg['long_mask'])

    test_dataset = LungDataset(dataset_path = os.path.join(cfg['main_path'], cfg['dataset']),
                                img_size = cfg['img_size'],
                                fold_cv = cfg['fold_cv'],
                                splitting_json = cfg['splitting'],
                                split = 'test',
                                joint_transform = tf_test,
                                one_hot_mask = False)

    testloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['workers'], pin_memory=True)
    logging.info(f"  - test dataset: {len(test_dataset)} samples")

    ## load model
    if cfg['model_name'] == 'UNet':
        model = UNet(in_channels=cfg['im_channels'], num_classes=cfg['num_classes'], base_filters=64, bilinear=True).to(device)

    checkpoint_file = os.path.join(checkpoints_path, f'{saved_model}_model.pth')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    ## loss function
    class_weights = torch.tensor(cfg['class_weights'], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ## evaluate
    # cfg['mode'] = test
    print('Evaluation...')
    dices, mean_dice, per_image_results, val_losses, dice_per_class = eval_mask(
        testloader, model, criterion=criterion, cfg=cfg, device=device
    )
    print(dices.shape)
    print()

    fold_metrics = {
        'mean_dice': float(mean_dice),
        'pleura_dice': float(dice_per_class[1]),
        'ribs_dice': float(dice_per_class[2]),
    }

    return fold_metrics, per_image_results, dices


def main(args):

    logging_dict = {'debug': logging.DEBUG, 'info': logging.INFO, 'warning': logging.WARNING,
                    'error': logging.ERROR, 'critical': logging.CRITICAL}
    logging.basicConfig(level=logging_dict[args.log])

    device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']

    all_results = {}       # per-image results across all folds
    fold_metrics = {}      # summary metrics per fold

    dices = []
    for fold in folds:
        logging.info(f"\n{'='*30}")
        logging.info(f" Testing {fold}...")
        logging.info(f"{'='*30}")

        metrics, per_image, dice_f = test_fold(
            fold       = fold,
            model_path = args.model_path,
            timestamp  = args.timestamp,
            data_path  = args.data_path,
            saved_model= args.saved_model,
            device     = device
        )

        fold_metrics[fold] = metrics
        all_results[fold]  = per_image
        dices.append(dice_f)

        logging.info(f"  mean dice     : {metrics['mean_dice']:.4f}")
        logging.info(f"  pleura dice   : {metrics['pleura_dice']:.4f}")
        logging.info(f"  ribs dice     : {metrics['ribs_dice']:.4f}")

    ## ── Summary ──────────────────────────────────────────────────────────────
    mean_dices   = [fold_metrics[f]['mean_dice']   for f in folds]
    pleura_dices = [fold_metrics[f]['pleura_dice'] for f in folds]
    ribs_dices   = [fold_metrics[f]['ribs_dice']   for f in folds]

    print('\n' + '='*40)
    print('  5-Fold Cross Validation Results')
    print('='*40)
    for fold in folds:
        m = fold_metrics[fold]
        print(f"  {fold}  |  mean: {m['mean_dice']:.4f}  |  pleura: {m['pleura_dice']:.4f}  |  ribs: {m['ribs_dice']:.4f}")
    print('-'*40)
    print(f"  mean dice   : {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
    print(f"  pleura dice : {np.mean(pleura_dices):.4f} ± {np.std(pleura_dices):.4f}")
    print(f"  ribs dice   : {np.mean(ribs_dices):.4f} ± {np.std(ribs_dices):.4f}")
    print('='*40)

    ## figure
    dices_all = np.concatenate(dices, axis=0)
    print("All dices shape:", dices_all.shape)
    pleura = dices_all[:,1]
    ribs   = dices_all[:,2]

    plt.figure(figsize=(6,5))

    plt.violinplot([pleura, ribs], showmedians=True)

    plt.xticks([1,2], ['Pleura','Ribs'])
    plt.ylabel('Dice score')
    plt.title('Dice distribution across 5 folds')

    plt.grid(alpha=0.3)

    fig_path = os.path.join(args.model_path, f'dice_violin_{args.timestamp}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    
    print(f"Violin plot saved to: {fig_path}")

    ## ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        'timestamp': args.timestamp,
        'saved_model': args.saved_model,
        'summary': {
            'mean_dice':   {'mean': float(np.mean(mean_dices)),   'std': float(np.std(mean_dices))},
            'pleura_dice': {'mean': float(np.mean(pleura_dices)), 'std': float(np.std(pleura_dices))},
            'ribs_dice':   {'mean': float(np.mean(ribs_dices)),   'std': float(np.std(ribs_dices))},
        },
        'fold_metrics': fold_metrics,
        'per_image_results': all_results,
    }

    save_path = os.path.join(args.model_path, f'test_5cv_{args.timestamp}.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"\n  Results saved to: {save_path}")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test UNet model - 5-fold cross validation')
    parser.add_argument('--log',        type=str, default='info',  help='Logging level')
    parser.add_argument('--model_path', type=str, required=True,   help='Path up to UNet folder (contains fold_1, fold_2, ...)')
    parser.add_argument('--timestamp',  type=str, required=True,   help='Shared timestamp across folds (e.g. 14-03-2026_11-27)')
    parser.add_argument('--saved_model',type=str, default='best',  help='Select best or last checkpoint')
    parser.add_argument('--data_path',  type=str, default=None,    help='Override main_path from train config (e.g. for local testing)')
    args = parser.parse_args()

    main(args)