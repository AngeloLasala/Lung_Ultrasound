"""
Dataset and data augmentation for 2D medical image segmentation.

Note: adapted form SAM
"""

import os
from random import randint
import numpy as np
import torch
from skimage import color
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from typing import Callable
import os
import cv2
import pandas as pd
import random
import json
import matplotlib.pyplot as plt

def to_long_tensor(pic):
    """
    Convert a PIL image or NumPy array into a PyTorch LongTensor.

    This function ensures that the input label or mask is converted 
    into a tensor of type torch.int64 (LongTensor), which is required 
    by PyTorch loss functions such as CrossEntropyLoss or NLLLoss 
    that expect class indices as integer labels (not one-hot or float tensors).

    Parameters:
    ----------
        pic (PIL.Image or numpy.ndarray): Input image or mask. 
            Typically a 2D array where each pixel represents a class index.

    Returns:
    --------
        torch.LongTensor: Tensor containing the same data as `pic`, 
        converted to integer (int64) type.
    """
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    """
    Function that uniforms the dimention of input images.
    Grey: (H, W)    -> (H, W, 1)
    RGB:  (H, W, 3) -> (H, W, 3) 

    Parameters:
    -----------
        images: multiple input of images

    Returns:
    -------
        corr_images: list of corrected images
    """
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)
    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class JointTransform2D:
    """
    Data augmentation on image and mask. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(self, 
                img_size=256, low_img_size=256, ori_size=256, crop=(32, 32), p_flip=0.0, p_rota=0.0, p_scale=0.0, p_gaussn=0.0, p_contr=0.0,
                p_gama=0.0, p_distor=0.0, color_jitter_params=(0.1, 0.1, 0.1, 0.1), p_random_affine=0,
                long_mask=False):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):

        #  gamma enhancement
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            # g = 2
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)

        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)
    
        # random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        # random horizontal flip
        if np.random.rand() < self.p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image, mask = F.rotate(image, angle), F.rotate(mask, angle)

        # random scale and center resize to the original size
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image, mask = F.resize(image, (new_h, new_w), InterpolationMode.BILINEAR), F.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)
            # image = F.center_crop(image, (self.img_size, self.img_size))
            # mask = F.center_crop(mask, (self.img_size, self.img_size))
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)

        # random add gaussian noise
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size)) * ns
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = F.to_pil_image(image.astype('uint8'))

        # random change the contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.5, 1.5))
            image = contr_tf(image)

        # random distortion
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)

        # color transforms,  ONLY ON IMAGE
        if self.color_jitter_params:
            image = self.color_tf(image)

        # random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params((-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image, mask = F.affine(image, *affine_params), F.affine(mask, *affine_params)
  
        # transforming to tensor
        image, mask = F.resize(image, (self.img_size, self.img_size), InterpolationMode.BILINEAR), F.resize(mask, (self.ori_size, self.ori_size), InterpolationMode.NEAREST)
        low_mask = F.resize(mask, (self.low_img_size, self.low_img_size), InterpolationMode.NEAREST)
    
        image = F.to_tensor(image)

        if not self.long_mask:
            mask = F.to_tensor(mask) * 255
            low_mask = F.to_tensor(low_mask) * 255
        else:
            mask = to_long_tensor(mask) * 255
            low_mask = to_long_tensor(low_mask) * 255
        
        return image, mask, low_mask

class LungDataset(Dataset):
    """
    Dataset class for lung segmentation of pleura and ribs'chadws. 

    The tree data is the follow
    ../../OpenPOCUS/Extrapolates_frames
    ├── patient_file_name_1
    │   ├── images
    │   |   ├── pt_zone_frame_0000.png
    │   |   ├── pt_zone_frame_0001.png
    │   ├── labels
    │   |   ├── pt_zone_frame_0000.npy
    │   |   ├── pt_zone_frame_0001.npy
    │   ├── 
    ├── patient_file_name_2
    ├── patient_file_name_3
    └── patient_file_name_n

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self,
                dataset_path: str,
                img_size: int,
                fold_cv : str,
                splitting_json : str,
                split: str, 
                joint_transform: Callable = None, 
                one_hot_mask: int = False) -> None:

        # dataset info
        self.dataset_path = dataset_path

        ## get the splitting of the dataset
        self.fold_cv = fold_cv  
        self.split = split
        with open(os.path.join(self.dataset_path, splitting_json), 'r') as file:
            self.splitting_dict = json.load(file)
        self.subjects_files = self.splitting_dict[self.fold_cv][self.split]

        ## img information
        self.img_size = img_size
        self.one_hot_mask = one_hot_mask
        
        self.data_dict = self.get_data_list()
        self.subject_list, self.images_list, self.labels_list, self.zones_list = self.data_dict['subjects'], self.data_dict['images'], self.data_dict['labels'], self.data_dict['zones']

        ## trasformation and augmentation
        if joint_transform is not None:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(self.data_dict['subjects'])

    def __getitem__(self, idx):
        
        image, mask, subject, zone = self.get_image_label(idx)
        
        ## correct dimensions if needed
        image, mask = correct_dims(image, mask)  

        ## data augmentation on the fly, TO UPDATE ...
        image, mask, low_mask = self.joint_transform(image, mask)
        # image, mask = self.joint_transform(image, mask)

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        # low_mask = low_mask.unsqueeze(0)
        # mask = mask.unsqueeze(0)
        return {
            'image': image,
            'label': mask,
            'subject': subject,
            'zone': zone 
            }


    def get_image_label(self, idx):
        """
        Get image and label
        """
        subject = self.subject_list[idx]
        img_path = self.images_list[idx]
        label_path = self.labels_list[idx]
        zone = self.zones_list[idx] 

        image = cv2.imread(img_path, 0)
        label = np.load(label_path)

        return  image, label, subject, zone
        

    def get_data_list(self):
        """
        From patients name get the data list
        Note: this function take both pre and post for unified training
        subject_i -> pre/subject_i and post/subject_i
        """
        data_dict = {'subjects': [], 'images':[], 'labels':[], 'zones':[]}
        for subject in self.subjects_files:
            images_path = os.path.join(self.dataset_path, subject, 'images')
            labels_path = os.path.join(self.dataset_path, subject, 'labels')
            images_list = os.listdir(images_path)
            labels_list = [os.path.splitext(f)[0] + '.npy' for f in images_list if f.endswith('.png')]
            
            for i,j in zip(images_list, labels_list):
                data_dict['subjects'].append(subject)
                data_dict['images'].append(os.path.join(images_path, i))
                data_dict['labels'].append(os.path.join(labels_path, j))
                data_dict['zones'].append(i.split('_')[1])

        return data_dict

if __name__ == '__main__':
    from lung_ultrasound.quality_model.cfg import cfg 
    import matplotlib.patches as patches

    ## aug settings
    low_image_size = cfg.low_img_size       ## the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS
    encoder_input_size = cfg.img_size       ## the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS
    crop = cfg.crop 
    p_flip = cfg.p_flip
    p_rota = cfg.p_rota
    p_scale = cfg.p_scale
    p_gaussn = cfg.p_gaussn
    p_contr = cfg.p_contr
    p_gama = cfg.p_gama
    p_distor = cfg.p_distor
    color_jitter_params = cfg.color_jitter_params
    p_random_affine = cfg.p_random_affine
    long_mask = cfg.long_mask

    transform = JointTransform2D(img_size=encoder_input_size, low_img_size=low_image_size, 
                                ori_size=cfg.img_size, 
                                crop=crop, 
                                p_flip=0, 
                                p_rota=0, 
                                p_scale=0,
                                p_gaussn=0,
                                p_contr=0, 
                                p_gama=0,
                                p_distor=0,
                                color_jitter_params = color_jitter_params,
                                long_mask=long_mask)                         

    dataset = LungDataset(dataset_path = os.path.join(cfg.main_path, cfg.dataset),
                          img_size = cfg.img_size,
                          fold_cv = cfg.fold_cv,
                          splitting_json = cfg.splitting,
                          split = 'train', 
                          joint_transform = transform, 
                          one_hot_mask = False)

    ## compute general information - size of 0 - 1 - 2
    ratios_list = []

    for i in range(len(dataset)):
        data = dataset[i]
        mask = data['label'].cpu().numpy().astype(np.int64)  # tensore -> numpy intero

        counts = np.bincount(mask.flatten(), minlength=3)  # conta pixel per classe 0,1,2
        total_pixels = mask.size  # attenzione: attributo, non funzione
        ratios = counts / total_pixels
        ratios_list.append(ratios)

    ratios_array = np.array(ratios_list)

    for cls in range(3):
        cls_ratios = ratios_array[:, cls]
        mean_val = np.mean(cls_ratios)
        median_val = np.median(cls_ratios)
        print(f"Classe {cls}: media={mean_val:.3f}, mediana={median_val:.3f}")

        plt.figure()
        plt.hist(cls_ratios, bins=20, color=['skyblue','lightgreen','salmon'][cls])
        plt.title(f"Istogramma ratio classe {cls}")
        plt.xlabel("Ratio")
        plt.ylabel("Numero di immagini")
        plt.grid(True)
        plt.show()
    
    idx = 10
    for i in range(10):
        data = dataset[idx]

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), num=i)
        axes[0].imshow(data['image'].permute(1,2,0).numpy(), cmap='gray')
        axes[0].set_title("Img")
        # axes[0].axis('off')

        axes[1].imshow(data['image'].permute(1,2,0).numpy(), cmap='gray')
        axes[1].imshow(data['label'][0], alpha=0.2, cmap='jet')
        axes[1].set_title("Img + mask")
        
        plt.show()
    