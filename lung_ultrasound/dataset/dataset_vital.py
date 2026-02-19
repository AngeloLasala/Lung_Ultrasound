"""
Dataset script for defining DatasetVital for temporal analysis compatible with the appl
"""
import os
import json
import torchvision
import torch.nn.functional as F
from PIL import Image
from typing import Optional
import logging
from dataclasses import dataclass, field
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import h5py
import numpy as np
import math
import matplotlib.pyplot as plt

@dataclass
class AugmentationConfig:
    """
    Holds all hyper-parameters that govern data augmentation.
    Pass an instance to DatasetVitalPOCUS; set data_augmentation=False
    to disable everything at inference time.

    Spatial (applied with the SAME random state across all frames in a segment
    to preserve temporal coherence):
        h_flip_p       : probability of horizontal flip
        v_flip_p       : probability of vertical flip
        rotation_deg   : max absolute rotation in degrees  (uniform sample)
        crop_scale     : (min, max) scale range for RandomResizedCrop        -> to deactive [1.0,1.0]
        crop_ratio     : (min, max) aspect-ratio range for RandomResizedCrop -> to deactive [1.0,1.0]

    Intensity (applied INDEPENDENTLY per frame to mimic realistic US artifacts):
        brightness     : max delta added to brightness  (uniform in [-v, v])
        contrast       : multiplicative contrast factor range (uniform in [1-v, 1+v])
        gamma_range    : (min, max) for random gamma correction
        gaussian_noise_std: std of additive Gaussian noise (0 → disabled)
        speckle_noise_std : std of multiplicative speckle noise (0 → disabled)
    """
    # Spatial
    h_flip_p: float = 0.5
    v_flip_p: float = 0.0
    rotation_deg: float = 23.0
    crop_scale: tuple = field(default_factory=lambda: (1.0, 1.0))
    crop_ratio: tuple = field(default_factory=lambda: (1.0, 1.0))

    # Intensity
    brightness_p: float = 0.5
    contrast_p: float = 0.5
    brightness: float = 0.10
    contrast: float = 0.10
    # gamma_range: tuple = field(default_factory=lambda: (0.8, 1.2))
    # gaussian_noise_std: float = 0.02
    # speckle_noise_std: float = 0.02

class VideoAugmentationPipeline:
    """
    Applies stochastic augmentations to an ultrasound video segment.

    Input / output shape: (T, H, W), float32, values in [0, 1].
    """

    def __init__(self, config: AugmentationConfig, target_size: tuple):
        self.cfg = config
        self.target_size = target_size  # (H, W)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        video : torch.Tensor, shape (T, H, W), float32, in [0, 1]

        Returns
        -------
        torch.Tensor, same shape and dtype, values clamped to [0, 1]
        """
        video = self._spatial_augment(video)    # (T, H, W)
        video = self._intensity_augment(video)  # (T, H, W)
        return video.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Spatial augmentations  (same random state for all frames)
    # ------------------------------------------------------------------
    def _spatial_augment(self, video: torch.Tensor) -> torch.Tensor:
        T, H, W = video.shape

        # --- sample random parameters once for the whole segment ---

        do_hflip = torch.rand(1).item() < self.cfg.h_flip_p
        do_vflip = torch.rand(1).item() < self.cfg.v_flip_p

        angle = (2 * torch.rand(1).item() - 1) * self.cfg.rotation_deg

        # Crop parameters (top, left, crop_h, crop_w)
        scale = self.cfg.crop_scale
        ratio = self.cfg.crop_ratio
        s = scale[0] + (scale[1] - scale[0]) * torch.rand(1).item()
        r_log_min = np.log(ratio[0])
        r_log_max = np.log(ratio[1])
        r = np.exp(r_log_min + (r_log_max - r_log_min) * torch.rand(1).item())
        crop_h = min(H, int(round(H * s / np.sqrt(r))))
        crop_w = min(W, int(round(W * s * np.sqrt(r))))
        top  = torch.randint(0, max(1, H - crop_h + 1), (1,)).item()
        left = torch.randint(0, max(1, W - crop_w + 1), (1,)).item()

        # --- apply to every frame ---
        frames = []
        for t in range(T):
            frame = video[t]  # (H, W)

            if do_hflip:
                frame = TF.hflip(frame.unsqueeze(0)).squeeze(0)
            if do_vflip:
                frame = TF.vflip(frame.unsqueeze(0)).squeeze(0)

            # Rotation + shear via affine (fill=0 = black, appropriate for US)
            frame = TF.affine(frame.unsqueeze(0),
                              angle=angle,
                              translate=[0, 0],
                              scale=1.0,
                              shear=[0.0, 0.0],
                              fill=0).squeeze(0)

            # Crop + resize back to target
            frame = TF.resized_crop(frame.unsqueeze(0),
                                    top=top, left=left,
                                    height=crop_h, width=crop_w,
                                    size=list(self.target_size),
                                    antialias=True).squeeze(0)

            frames.append(frame)

        return torch.stack(frames, dim=0)   # (T, H, W)


    def _intensity_augment(self, video: torch.Tensor) -> torch.Tensor:

        do_brightness = torch.rand(1).item() < self.cfg.brightness_p
        do_contrast   = torch.rand(1).item() < self.cfg.contrast_p

        delta = (2 * torch.rand(1).item() - 1) * self.cfg.brightness
        factor = 1.0 + (2 * torch.rand(1).item() - 1) * self.cfg.contrast

        T = video.shape[0]
        frames = []
        for t in range(T):
            frame = video[t].unsqueeze(0)   # (1, H, W)

            # Brightness shift
            if self.cfg.brightness > 0 and do_brightness:
                frame = frame + delta

            # Contrast scaling around mean
            if self.cfg.contrast > 0 and do_contrast:
                mean = frame.mean()
                frame = (frame - mean) * factor + mean

            # Gamma correction
            # lo, hi = self.cfg.gamma_range
            # gamma = lo + (hi - lo) * torch.rand(1).item()
            # frame = frame.clamp(0.0, 1.0).pow(gamma)

            # # Additive Gaussian noise
            # if self.cfg.gaussian_noise_std > 0:
            #     frame = frame + torch.randn_like(frame) * self.cfg.gaussian_noise_std

            # # Multiplicative speckle noise  (characteristic of ultrasound)
            # if self.cfg.speckle_noise_std > 0:
            #     frame = frame * (1.0 + torch.randn_like(frame) * self.cfg.speckle_noise_std)

            frames.append(frame.squeeze(0))

        return torch.stack(frames, dim=0)   # (T, H, W)

class DatasetVital(torch.utils.data.Dataset):
    """
    DatasetVital class for developing DL model for LUS video analysis.
    Dowstream task is the classification of the multual-exslivide bio-markers:
    ["A-lines", "B-lines", "Confluent B-line", "Consolidation", "Effussion"]

    form more details: https://github.com/vital-ultrasound/public-lung
    ---------------------------
    The tree data is the follow
    ../../DATA_covid
    ├── video_file_name_1
    │   ├── images
    │   |   ├── frame_0000.png
    │   │   ├── frame_0003.png
    │   │   ├── labels
    │   |   ├── label_info.txt 
    ├── video_file_name_2
    ├── video_file_name_3
    └── video_file_name_n
    """
    def __init__(self, dataset_path,
                      data_augmentation,
                      size,
                      im_channels,
                      lenght,
                      overlap,
                      fps,    
                      sampling_f,
                      splitting_json,
                      fold_cv,
                      split, 
                      trasformations) :
        # dataset info
        self.dataset_path = dataset_path

        # condition and data augmentation parameters
        self.data_augmentation = data_augmentation
        
        # video parameters
        self.size = size
        self.im_channels = im_channels
        self.lenght = lenght
        self.overlap = overlap
        self.fps = fps
        self.sampling_f = sampling_f

        ## get the splitting of the dataset
        self.fold_cv = fold_cv  
        self.split = split
        with open(os.path.join(self.dataset_path, splitting_json), 'r') as file:
            self.splitting_dict = json.load(file)

            
        self.subjects_files = self.splitting_dict[self.fold_cv][self.split]

        ## frames and label list
        dict_videos_labels = self._get_videos_label_dict()
        self.videos_list, self.labels_list, self.subject_list = dict_videos_labels['videos'], dict_videos_labels['labels'], dict_videos_labels['subjects']

        self.trasformations = trasformations

    def __len__(self):
        return len(self.videos_list)

    def __getitem__(self, index):
        video, label, subject = self.get_video_label(index)

        if self.trasformations is not None:
            video_tensor = torch.tensor(video, dtype=torch.float32)  # (30, H, W)
            video_tensor = video_tensor.unsqueeze(1)  # (30, 1, H, W)
            video_tensor = F.interpolate(
                video_tensor,
                size=self.size,
                mode="bilinear",
                align_corners=False
            )
            # reshape to (C, T, H, W)
            video_tensor = video_tensor.permute(1, 0, 2, 3)  # (1, 30, H, W)
            video_tensor = video_tensor / 255.0  

            label_tensor = torch.tensor(label, dtype=torch.float32)

        else:
            video_tensor = torch.tensor(video, dtype=torch.float32).unsqueeze(0)/255.
            label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return video_tensor, label_tensor, subject

        
    def get_video_label(self, index):
        """
        Return the image and label given the index
        """
        video = self.videos_list[index]

        subject = self.subject_list[index]

        # read json as a dictionary
        with open(self.labels_list[index], 'r') as f:
            label = json.load(f)
        markers = ["A-lines", "B-lines", "Consolidations", "Effusion"]
        classes = [0, 1, 3, 4]  # compatible vith vital

        bio_markers_one_hot = np.zeros(5, dtype=int)
        bio_markers_one_hot[classes] = [label[m] for m in markers]
        
        return video, bio_markers_one_hot, subject
        
    
    def _get_videos_label_dict(self):
        """
        From the list of patient in self.subjects_files return the list of videos and label
        """
        image_label_dict = {'videos': [], 'labels': [], 'subjects': []}
        for subject in self.subjects_files:
            subject_path = os.path.join(self.dataset_path, subject)

            video_frames_path = os.path.join(subject_path, 'images')
            video_labels_path = os.path.join(subject_path, 'labels', 'label_info.json')

            frames = self._get_sequences_of_frames(video_frames_path)
            for i in frames:
                image_label_dict['videos'].append(i)
                image_label_dict['labels'].append(video_labels_path)
                image_label_dict['subjects'].append(subject)
            
        return image_label_dict

    def _get_sequences_of_frames(self, video_path):
        """
        From a folder with frames, retarn a list of segments of frames with the specified length and overlap.

        Parameters
        ----------
        video_path: str
            Path to the folder with the frames of the video. The frames should be named as "frame_00001.png", "frame_00002.png", etc.
        lenght: int
            Length of the segments of frames to return (in seconds)
        overlap: float
            Overlap between segments (percentage between 0 and 1)
        size: tuple
            Size of the frames to return (width, height)
        fps: int
            Frames per second of the video (default: 30)
        sampling_f: int
            Sampling frequency of the frames (default: 30)
        """
        total_video_frames = len(os.listdir(video_path))
        segment_length_frames = int(self.lenght * self.sampling_f)
        step_frames = int(segment_length_frames * (1 - self.overlap))

        segments = []
        for start in range(0, total_video_frames - segment_length_frames + 1, step_frames):
            end = start + segment_length_frames
            segments.append((start, end))

        frames = []
        for start, end in segments:
            segment_frames = []
            for frame_idx in range(start, end):
                frame_path = os.path.join(video_path, f"frame_{frame_idx:04d}.png")
                if os.path.exists(frame_path):
                    frame = Image.open(frame_path)
                    frame = (np.array(frame))#/ 255.0
                    segment_frames.append(frame)
                else:
                    pass
                    segment_frames.append(torch.zeros((frame.shape[0], frame.shape[1])))  # Placeholder for missing frames
            segment_frames = np.stack(segment_frames, axis=0)
            frames.append(segment_frames)

        return frames

class DatasetVitalPOCUS(torch.utils.data.Dataset):
    """
    DatasetVital class for developing DL model for LUS video analysis.
    Dowstream task is the classification of the multual-exslivide bio-markers:
    ["Normal", "B-lines", "Consolidation", "B-lines + Consolidations", "Indeterminate"]

    Note: I suppose that the acquisitions fps is 30 fps, so I adopt the same structure of above class

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
    def __init__(self, dataset_path : str,
                      size : tuple,
                      im_channels : int,
                      lenght : int,
                      overlap : float,
                      fps : int,    
                      sampling_f : int,
                      splitting_json : str,
                      fold_cv : str,
                      split : str,
                      data_augmentation : bool,
                      normalize : bool,
                      aug_config : Optional[AugmentationConfig] = None,
                      mean : Optional[float] = None, 
                      std : Optional[float] = None) :
        # dataset info
        self.dataset_path = dataset_path
        
        # video parameters
        self.size = size
        self.im_channels = im_channels
        self.lenght = lenght
        self.overlap = overlap
        self.fps = fps
        self.sampling_f = sampling_f

        ## get the splitting of the dataset
        self.fold_cv = fold_cv  
        self.split = split
        with open(os.path.join(self.dataset_path, splitting_json), 'r') as file:
            self.splitting_dict = json.load(file)

        # data augmentation parameters
        self.data_augmentation = data_augmentation
        self.aug_config = aug_config is not None
        self.aug_pipeline = VideoAugmentationPipeline(aug_config, target_size=self.size) 
        self.normalize = normalize
        

        # zones list - double format to deal witht he incopatability between folder and labels keys
        self.zones = {'z1':'z01','z2':'z02','z3':'z03','z4':'z04','z5':'z05',
                    'z6':'z06','z7':'z07','z8':'z08','z9':'z09','z10':'z10','z11':'z11','z12':'z12'} 
        self.subjects_files = self.splitting_dict[self.fold_cv][self.split]

        ## frames and label list
        dict_videos_labels = self._get_videos_label_dict()
        self.videos_list, self.labels_list, self.subject_list, self.zones_list = dict_videos_labels['videos'], dict_videos_labels['labels'], dict_videos_labels['subjects'], dict_videos_labels['zones']
        
        self.mean, self.std = self.compute_dataset_stats().values() if split != 'test'  else (mean, std)  # default to 0.5 if not training split


    def __len__(self):
        return len(self.videos_list)

    def __getitem__(self, index):
        video, label, subject, zone = self.get_video_label(index)
        video_tensor = torch.tensor(video, dtype=torch.float32) / 255.0  # (T, H, W)
        label_tensor = torch.tensor(label)
        
        if self.data_augmentation:   ## SPATIAL AND INTENSITY AUGMENTATION
            video_tensor = self.aug_pipeline(video_tensor)  # (T, H, W)
       
        if self.normalize:
            mean = [self.mean] * video_tensor.shape[0]
            std = [self.std] * video_tensor.shape[0]
            video_tensor = transforms.Normalize(mean=mean, std=std)(video_tensor)  # Normalize to estimated dataset mean and std
            # video_tensor = trasforms.Normalize(mean=[0.5], std=[0.5])(video_tensor)  # Normalize to [-1, 1]
            # video_tensor = trasforms.Normalize(mean=[0.485], std=[0.229])(video_tensor)  # Normalize to Imagenet mean and std


        video_tensor = video_tensor.unsqueeze(0)
        
        return video_tensor, label_tensor, subject, zone
        
    def get_video_label(self, index):
        """
        Return the image and label given the index
        """
        video = self.videos_list[index]
        subject = self.subject_list[index]
        label = self.labels_list[index]
        zone = self.zones_list[index]

        return video, label, subject, zone

    def _get_videos_label_dict(self):
        """
        From the list of patient in self.subjects_files return the list of videos and label
        """
        image_label_dict = {'videos': [], 'labels': [], 'subjects': [], 'zones': []}
        for subject in self.subjects_files:
            subject_path = os.path.join(self.dataset_path, subject)

            ## read label json
            subject_labels_path = os.path.join(subject_path, 'labels.json')
            subject_labels = json.load(open(subject_labels_path, 'r'))

            for zone in self.zones.keys():
                if subject_labels[zone] != 'Nan':
                    video_frames_path = os.path.join(subject_path, self.zones[zone], f"{self.zones[zone]}.h5")

                    with h5py.File(video_frames_path, "r") as f:
                        video_df = f["images"]
                        video_frames = video_df[:]

                    frames = self._get_sequences_of_frames(video_frames)
                    for i in frames:
                        image_label_dict['videos'].append(i)
                        image_label_dict['labels'].append(subject_labels[zone])
                        image_label_dict['subjects'].append(subject)
                        image_label_dict['zones'].append(zone)
                
        return image_label_dict

    def _get_sequences_of_frames(self, video_frames):
        """
        Divide the video in segments of frames with the specified length and overlap.

        Parameters
        ----------
        video_tensor : np.ndarray or torch.Tensor

        Returns
        -------
        frames : list
            Listod segments, shape (segment_length_frames, H, W)
        """
        total_video_frames = video_frames.shape[0]
        segment_length_frames = int(self.lenght * self.sampling_f)
        step_frames = int(segment_length_frames * (1 - self.overlap))

        segments = []
        for start in range(0, total_video_frames - segment_length_frames + 1, step_frames):
            end = start + segment_length_frames
            segments.append((start, end))

        frames = []
        for start, end in segments:
            segment_frames = video_frames[start:end]
            frames.append(segment_frames)

        return frames

    def compute_dataset_stats(self, max_samples= 500):
        """
        Estimate the dataset-level mean and standard deviation by sampling
        up to *max_samples* segments at random (without replacement if the
        dataset is smaller).

        Call this once on the **training split**, save the values, and pass
        them to NormalizationConfig for all splits:

            stats = train_ds.compute_dataset_stats()
            norm_cfg = NormalizationConfig(mode="dataset",
                                           mean=stats["mean"],
                                           std=stats["std"])

        Returns
        -------
        dict with keys "mean" and "std" (Python floats, values over [0,1] pixels)
        """
        n = len(self.videos_list)
        indices = torch.randperm(len(self.videos_list))[:n].tolist()

        running_sum   = 0.0
        running_sum_sq = 0.0
        running_count  = 0

        for idx in indices:
            # raw segment as float in [0,1], no augmentation
            seg = torch.tensor(self.videos_list[idx], dtype=torch.float32) / 255.0
            running_sum    += seg.sum().item()
            running_sum_sq += (seg ** 2).sum().item()
            running_count  += seg.numel()

        mean = running_sum / running_count
        std  = ((running_sum_sq / running_count) - mean ** 2) ** 0.5

        return {"mean": mean, "std": std}

if __name__ == "__main__":
    ## PLAYGRAOUDN

    ## Dataset vital
    dataset_path = '/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/DATA_covid_compvital'
    size = (64,64)      # size of each frame
    im_channels = 1     # number of channels of the image (1 for grayscale, 3 for RGB)
    fold_cv = 'fold_1'  # numer of the 5-fold cross validation fold to use (from fold_1 to fold_5)   
    split = 'test'
    lenght = 1         # lenght of the segments of frames to return (in seconds)
    overlap = 0.2      # overlap between segments (percentage between 0 and 1)
    sampling_f = 30      # sampling frequency of the frames (default: 30)
    fps = 30


    # dataset = DatasetVital(dataset_path = dataset_path,
    #                         data_augmentation = False,
    #                         size = size,
    #                         im_channels = im_channels,
    #                         lenght = lenght,
    #                         overlap = overlap,
    #                         fps = fps,
    #                         sampling_f = sampling_f,
    #                         splitting_json = 'splitting.json',
    #                         fold_cv = fold_cv,
    #                         split = split, 
    #                         trasformations = transforms)
    
    # print(f"Dataset length: {len(dataset)}")
    # print(f"Video shape: {dataset[0][0].shape}", dataset[0][0].min(), dataset[0][0].max())  

    # for img, label, subject in dataset:
    #     print(f"Subject: {subject}, image: {img.shape}, ")

        ## plot first last and middle frame
        # plt.figure(figsize=(15,5))
        # plt.subplot(1,3,1)
        # plt.imshow(img[0,0,:,:], cmap='gray')
        # plt.title('First frame')
        # plt.subplot(1,3,2)
        # plt.imshow(img[0,-1,:,:], cmap='gray')
        # plt.title('Last frame')
        # plt.subplot(1,3,3)
        # plt.imshow(img[0,img.shape[1]//2,:,:], cmap='gray')
        # plt.title('Middle frame')
        # plt.show()

    ## Dataset vital pocus
    dataset_path = "/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/DATA_Lung_Database"
    size = (64*2,64*2)      # size of each frame
    im_channels = 1     # number of channels of the image (1 for grayscale, 3 for RGB)
    fold_cv = 'fold_1'  # numer of the 5-fold cross validation fold to use (from fold_1 to fold_5)   
    split = 'test'
    splitting = 'splitting.json'
    lenght = 1         # lenght of the segments of frames to return (in seconds)
    overlap = 0.2      # overlap between segments (percentage between 0 and 1)
    sampling_f = 30    # sampling frequency of the frames (default: 30)
    fps = 30

    # give me augmentation with all deactivation parameters
    aug_config = AugmentationConfig(h_flip_p=0,
                                    v_flip_p=0.0,
                                    rotation_deg=0.0,
                                    crop_scale=(1.0, 1.0),
                                    crop_ratio=(1.0, 1.0),
                                    brightness_p=0.0,
                                    contrast_p=0.0,
                                    brightness=0.0,
                                    contrast=0.0
                                    )

    print("Creating DatasetVitalPOCUS...")
    dataset = DatasetVitalPOCUS(dataset_path = dataset_path,
                            size = size,
                            im_channels = im_channels,
                            lenght = lenght,
                            overlap = overlap,
                            fps = fps,
                            sampling_f = sampling_f,
                            splitting_json = splitting,
                            fold_cv = fold_cv,
                            split = split, 
                            normalize = True,
                            data_augmentation = True,
                            aug_config = aug_config)
    print("DatasetVitalPOCUS created.")
    print(f"Dataset length: {len(dataset)}")
    print(f"Estimated dataset mean: {dataset.mean:.4f}, std: {dataset.std:.4f}")


    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print(dataset.data_augmentation)
    print(dataset.aug_config)

    for img, label, subject, zone in dataset:
        # plot first last and middle frame
        print(img.shape)
        print(img.min(), img.max())

        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(img[0,0,:,:], cmap='gray')
        plt.title(f'Subject: {subject}, Zone: {zone}, Label: {label}, First frame')
        plt.subplot(1,3,2)
        plt.imshow(img[0,-1,:,:], cmap='gray')
        plt.title('Last frame')
        plt.subplot(1,3,3)
        plt.imshow(img[0,img.shape[1]//2,:,:], cmap='gray')
        plt.title('Middle frame')
        plt.show()



