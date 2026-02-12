"""
Dataset script for defining DatasetVital for temporal analysis
"""
import os
import json
import torchvision
import torch.nn.functional as F
from PIL import Image
import logging
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import math
import matplotlib.pyplot as plt

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
        From the list of patient in self.subjects_files return the list of image and tumor
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


if __name__ == "__main__":
    # PLAYGRAOUDN
    dataset_path = '/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/DATA_covid_compvital'
    size = (64,64)      # size of each frame
    im_channels = 1     # number of channels of the image (1 for grayscale, 3 for RGB)
    fold_cv = 'fold_1'  # numer of the 5-fold cross validation fold to use (from fold_1 to fold_5)   
    split = 'test'
    lenght = 1         # lenght of the segments of frames to return (in seconds)
    overlap = 0.2      # overlap between segments (percentage between 0 and 1)
    sampling_f = 30      # sampling frequency of the frames (default: 30)
    fps = 30


    dataset = DatasetVital(dataset_path = dataset_path,
                            data_augmentation = False,
                            size = size,
                            im_channels = im_channels,
                            lenght = lenght,
                            overlap = overlap,
                            fps = fps,
                            sampling_f = sampling_f,
                            splitting_json = 'splitting.json',
                            fold_cv = fold_cv,
                            split = split, 
                            trasformations = transforms)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Video shape: {dataset[0][0].shape}", dataset[0][0].min(), dataset[0][0].max())  

    for img, label, subject in dataset:
        print(f"Subject: {subject}, Label: {label}, ")

    


