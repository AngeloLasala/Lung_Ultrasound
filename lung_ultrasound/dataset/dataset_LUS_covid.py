"""
Create the DatasetLUSCovid compatible with the Pythorch framework
and the main visualiztion dataset statistics
"""
import os
import json
import torchvision
from PIL import Image
from tqdm import tqdm
import json
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import numpy as np
import math
import matplotlib.pyplot as plt

class DatasetLUSCovid(torch.utils.data.Dataset):
    """
    DatasetLUSCovid class for developing DL model for LUS analysis

    Important considerations: for multi instance classification, biomarkers are in this order
    ["Effusion", "Consolidations", "B-lines", "A-lines", "Pleural line irregularities", "Air bronchogram"]

    ---------------------------
    The tree data is the follow
    ../../DATA_covid
    ├── video_file_name_1
    │   ├── images
    │   |   ├── frame_0000.png
    │   │   ├── frame_0003.png
        │   ├── labels
    │   |   ├── label_info.txt 
    ├── Case2-US-before
    ├── Case3-US-before
    └── Case4-US-before
    """
    def __init__(self, dataset_path,
                      data_augmentation,
                      size,
                      im_channels,
                      splitting_json,
                      fold_cv,
                      split) :
        self.dataset_path = dataset_path

        # condition and data augmentation parameters
        self.data_augmentation = data_augmentation
        
        #img parameters
        self.size = size
        self.im_channels = im_channels

        ## get the splitting of the dataset
        self.fold_cv = fold_cv  
        self.split = split
        with open(os.path.join(self.dataset_path, splitting_json), 'r') as file:
            self.splitting_dict = json.load(file)

        # for key in self.splitting_dict.keys():
        #     print(f"{key} :")
        #     for subkey in self.splitting_dict[key].keys():
        #         print(f"    {subkey}: {len(self.splitting_dict[key][subkey])} subjects")
            
        self.subjects_files = self.splitting_dict[self.fold_cv][self.split]

        ## image and label list
        self.image_list, self.label_list = self.get_image_label_dict()['images'], self.get_image_label_dict()['labels']

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im, label, subject = self.get_image_label(index)

        ################ IMAGE CONDITION ###################################
        if self.data_augmentation:
            im_tensor, labels_tensor = self.augmentation(im, label)
        else:
            im_tensor, labels_tensor = self.trasform(im, label)

        return im_tensor, labels_tensor, subject
        

    def get_image_label(self, index):
        """
        Return the image and label given the index
        """
        # read image and label with PIL
        im = Image.open(self.image_list[index])
        im = im.convert('RGB')

        subject = self.image_list[index].split('/')[-3]

        # read json as a dictionary
        with open(self.label_list[index], 'r') as f:
            label = json.load(f)
        bio_markers = [int(label[marker]) for marker in ["Effusion", "Consolidations", "B-lines", "A-lines", "Pleural line irregularities", "Air bronchogram"]]
        bio_markers = np.array(bio_markers)

        
        return im, bio_markers, subject
        
    
    def get_image_label_dict(self):
        """
        From the list of patient in self.subjects_files return the list of image and tumor
        """
        image_label_dict = {'images': [], 'labels': []}
        for subject in self.subjects_files:
            subject_path = os.path.join(self.dataset_path, subject)
            for item in os.listdir(os.path.join(self.dataset_path, subject, 'images')):
                image_label_dict['images'].append(os.path.join(subject_path, 'images', item))
                image_label_dict['labels'].append(os.path.join(subject_path, 'labels', 'label_info.json'))
                
        return image_label_dict

    def augmentation(self, image, label=None):
        """
        Set of trasformation to apply to image.
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)
        
        ## random rotation to image and label
        if torch.rand(1) > 0.5:
            angle = np.random.randint(-15, 15)
            image = transforms.functional.rotate(image, angle)

        ## random translation to image and label in each direction
        if torch.rand(1) > 0.5:
            translate = transforms.RandomAffine.get_params(degrees=(0.,0.), 
                                                        translate=(0.10, 0.10),
                                                        scale_ranges=(1.0,1.0),
                                                        shears=(0.,0.), 
                                                        img_size=self.size)
            image = transforms.functional.affine(image, *translate)

        ## random horizontal flip
        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)

        ## random vertical flip
        # if torch.rand(1) > 0.5:
        #     image = transforms.functional.vflip(image)
            
        ## random brightness and contrast
        if torch.rand(1) > 0.5:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5)(image)

        
        ## random gamma correction
        if torch.rand(1) > 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            image = transforms.functional.adjust_gamma(image, gamma)
      
        label = torch.tensor(label, dtype=torch.float32)
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1  

        return image, label

    def trasform(self, image, label):
        """
        Simple trasformaztion of the label and image. Resize and normalize the image and resize the label
        """
        ## Resize
        resize = transforms.Resize(size=self.size)
        image = resize(image)

        ## convert to tensor and normalize
        label = torch.tensor(label, dtype=torch.float32)
        image = transforms.functional.to_tensor(image)
        image = (2 * image) - 1    
        return image, label

if __name__ == "__main__":
    ## PLAYGRAOUND and DATASET DISTRIBUTION INFO 
    dataset_path='/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/DATA_covid'
    size=(256,256)
    im_channels=3
    fold_cv='fold_1'
    split='train'

    dataset_list = []
    for split in ['train', 'val', 'test']:
        dataset_split = DatasetLUSCovid(dataset_path=dataset_path,
                                         data_augmentation=False,
                                         size=size,
                                         im_channels=im_channels,
                                         splitting_json='splitting.json',
                                         fold_cv=fold_cv,
                                         split=split
                                        )
        print(len(dataset_split))
        dataset_list.append((dataset_split))

    # concatenate all dataset splits
    full_dataset = torch.utils.data.ConcatDataset(dataset_list)

    labels_count = np.zeros((6,))
    for image, label, subject in tqdm(full_dataset):
        labels_count += label.numpy()
    
    print("Dataset distribution:")
    biomarkers = ["Effusion", "Consolidations", "B-lines", "A-lines", "Pleural line irregularities", "Air bronchogram"]
    for i, biomarker in enumerate(biomarkers):
        print(f"{biomarker}: {labels_count[i]} samples")

    # plot pie chart (grafico a torta)
    plt.figure(figsize=(8, 8))
    plt.pie(
        labels_count,
        labels=biomarkers,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Dataset distribution of LUS COVID-19 Dataset")
    plt.axis('equal')  # rende la torta circolare
    plt.show()

    