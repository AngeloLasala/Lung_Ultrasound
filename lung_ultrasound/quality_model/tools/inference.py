"""
Inference script that take as input a Patient folder and returns the predicted Pleura and reabs for each frames
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode

import time
from tqdm import tqdm
import random
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2

from lung_ultrasound.quality_model.cfg import cfg
from lung_ultrasound.quality_model.models.unet import UNet
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.evaluation import eval_mask
from lung_ultrasound.quality_model.utils.visualization import visualize_inference, make_gif, plot_centroids_over_time

class PatientClass(Dataset):
    """
    Costum dataset class that return each frames of a videos in a readible dataset for inference

     form more details: https://github.com/kumarandre/OpenPOCUS 
    ---------------------------
    The tree data is the follow
    ../../DATA_Lung_Database
    ├── patient_file_name_1
    │   ├── labels.json    
    │   ├── z01
    │   |   ├── z01.h5 (if present)
    │   ├── z02
    │   |   ├── z02.h5 (if present)
    │   ├── z03
    │   ├── z04
    │   ├── z05
    │   ├── z06
    │   ├── z07
    │   ├── z08
    │   ├── z09
    │   ├── z10
    │   ├── z11
    │   ├── z12
    ├── patient_file_name_2
    ├── patient_file_name_3
    └── patient_file_name_n
    """
    def __init__(self,
                subject_path: str,
                img_size: int,
                one_hot_mask: int = False):

        self.subject_path = subject_path
        self.img_size = img_size
        self.zones = {'z1':'z01','z2':'z02','z3':'z03','z4':'z04','z5':'z05',
                    'z6':'z06','z7':'z07','z8':'z08','z9':'z09','z10':'z10','z11':'z11','z12':'z12'}

        image_label_dict = self.get_videos()
        self.videos, self.labels, self.zones = image_label_dict['videos'], image_label_dict['labels'], image_label_dict['zones'] 

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video, label, zone = self.videos[idx], self.labels[idx], self.zones[idx]

        video = self.video_preprocessing(video) # (F, C, H, W)
        
        return video, label, zone
        
    def video_preprocessing(self, video):
        """
        Preprocessing video frame by frame to get compatible with segmentation preprocessing
        """
        frames = []
        for frame in video:
            frame = F.to_pil_image(frame)
            frame = F.resize(frame, (self.img_size, self.img_size), InterpolationMode.BILINEAR)
            frames.append(F.to_tensor(frame))
        return torch.stack(frames)


    def get_videos(self):
        """
        get videos from path
        """
        image_label_dict = {'videos': [], 'labels': [], 'zones': []}
        

        ## read label json
        subject_labels_path = os.path.join(self.subject_path, 'labels.json')
        subject_labels = json.load(open(subject_labels_path, 'r'))

        for zone in self.zones.keys():
            if subject_labels[zone] != 'Nan':
                # --- Try .h5 first, then .mp4, then .avi ---
                h5_path  = os.path.join(self.subject_path, self.zones[zone], f"{self.zones[zone]}.h5")
                mp4_path = os.path.join(self.subject_path, self.zones[zone], f"{self.zones[zone]}.mp4")
                avi_path = os.path.join(self.subject_path, self.zones[zone], f"{self.zones[zone]}.avi")
                print(mp4_path)
                print(os.listdir(os.path.join(self.subject_path, self.zones[zone])))

                if os.path.exists(h5_path):
                    with h5py.File(h5_path, "r") as f:
                        video_frames = f["images"][:]  # (F, H, W)

                elif os.path.exists(mp4_path):
                    video_frames = self._load_mp4(mp4_path)  # (F, H, W)

                elif os.path.exists(avi_path):
                    video_frames = self._load_avi(avi_path)  # (F, H, W)

                else:
                    print(f"[WARNING] No video file found for zone '{zone}' — skipping.")
                    continue

                image_label_dict['videos'].append(video_frames)
                image_label_dict['labels'].append(subject_labels[zone])
                image_label_dict['zones'].append(zone)
                
        return image_label_dict
    
    def _load_mp4(self, mp4_path):
        """
        Load an mp4 video and convert frames to grayscale numpy array (F, H, W).
        
        Args:
            mp4_path: full path to the .mp4 file
        
        Returns:
            numpy array of shape (F, H, W) with uint8 values
        """
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {mp4_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # (H, W)
            frames.append(gray)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from: {mp4_path}")

        return np.stack(frames, axis=0)  # (F, H, W)
    
    def _load_avi(self, path: str) -> np.ndarray:
        """
        Load an .avi video and return frames as a numpy array (F, H, W).
        Converts to grayscale to match the .h5 / .mp4 pipeline.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open AVI file: {path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)

        cap.release()

        if not frames:
            raise ValueError(f"No frames read from AVI file: {path}")

        return np.array(frames)  # (F, H, W)

def main(args):
    """
    Inference
    """
    ## load model
    checkpoints_path = os.path.join(args.model_path, 'checkpoints')
    cfg_train_path = os.path.join(args.model_path, 'train_config.json')

    ## read cfg_train configuration file
    if not os.path.exists(cfg_train_path):
        raise FileNotFoundError(f"File not found: {cfg_train_path}")

    with open(cfg_train_path, 'r') as f:
        cfg = json.load(f)

    device = cfg['device']

    if cfg['model_name'] == 'UNet':
        model  = UNet(in_channels=cfg['im_channels'], num_classes=cfg['num_classes'], base_filters=64, bilinear=True).to(device)
    best_checkpoint = os.path.join(checkpoints_path, 'best_model.pth')

    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {best_checkpoint}")
    print()
    print('Creating model...')
    print(f"  Model: {cfg['model_name']}")
    print(f"  Loading checkpoint from {best_checkpoint}")
    print()


    checkpoint = torch.load(best_checkpoint, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()  # Set model to evaluation mode

    ## load data
    print("Load Patient videos")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Patient: {args.subject}")
    
    subject_path = os.path.join(args.dataset_path, args.subject)

    patient_dataset = PatientClass(subject_path = subject_path,
                                   img_size = cfg['img_size'])

    ## create save folder
    inference_folder = os.path.join(os.path.dirname(args.dataset_path), 'inference_segmentation')
    os.makedirs(inference_folder, exist_ok=True)

    subject_folder = os.path.join(inference_folder, args.subject)
    os.makedirs(subject_folder, exist_ok=True)

    ## Inference for each zone
    ff = 15
    for i in range(len(patient_dataset)):
        video, label, zone = patient_dataset[i]
        video = video.to(device)

        total_frames = video.shape[0]
        all_frames_dict = {
            "frame": [],
            "frame_pleura": [],
            "frame_ribs": [],
            "centroid_pleura": [],
            "centroid_ribs": [],
        }

        # Processa a batch di ff frames
        for start in range(0, total_frames, ff):
            end = min(start + ff, total_frames)
            batch = video[start:end]

            time_start = time.time()
            print(f'Batch [{start}:{end}] - start inference...')
            pred = model(batch)
            time_stop = time.time()
            print(f'inference time: {time_stop - time_start:.4f} s\n')

            batch_dict = visualize_inference(batch, pred)

            # Accumula i frame nel dict globale
            for key in all_frames_dict:
                all_frames_dict[key].extend(batch_dict[key])

        # Crea la GIF finale per questa zona
        gif_name = f"{args.subject}_{zone}_label_{label}.gif"
        gif_path = os.path.join(subject_folder, gif_name)
        make_gif(all_frames_dict, output_path=gif_path, fps=10)

        # Salva il plot dei centroidi per questa zona
        plot_name = f"{args.subject}_{zone}_label_{label}_centroids.png"
        fig = plot_centroids_over_time(all_frames_dict, fps=30, save_path=subject_folder, filename=plot_name)
        plt.close(fig)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet model for semantic segmentation')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--dataset_path', type=str, help='Path to the videos dataset')
    parser.add_argument('--subject', type=str, help='Subject identifier')
    args = parser.parse_args()

    main(args)
    
