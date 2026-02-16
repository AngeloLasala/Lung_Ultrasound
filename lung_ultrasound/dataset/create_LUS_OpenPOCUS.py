"""
Create the dataset from OpenPOCUS data 

for details see:
"KUMAR, Andre, et al. Creation of an Open-Access Lung Ultrasound Image Database 
For Deep Learning and Neural Network Applications. medRxiv, 2025, 2025.05. 09.25327337."
"""
import argparse
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import tqdm
import shutil
import zipfile
import h5py

def create_and_fill_zone_folder(video_folder, output_folder, subject_id, zones):
    """
    Create subject folder and zones subfodler
    """
    subject_path = os.path.join(output_folder, subject_id)
    os.makedirs(subject_path, exist_ok=True)

    zone_dict = {zone: [] for zone in zones}

    for zone in zones:
        zone_path = os.path.join(subject_path, zone)
        os.makedirs(zone_path, exist_ok=True)

    for i in os.listdir(video_folder):
        
        # check if i is an images 
        if i.endswith(".png") or i.endswith(".jpg") or i.endswith(".jpeg"):
            i_list = i.split("_")
            i_set = {x.lower() for x in i_list}
            zone_i = [zone for zone in zones if zone in i_set][0]

            image_path = os.path.join(video_folder, i)

            ## read and resize the image to 512x512
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            size = (128, 128)
            image = cv2.resize(image, size)
        
            # save image to the corresponding zone folder
            # cv2.imwrite(os.path.join(subject_path, zone_i, i), image)

            zone_dict[zone_i].append(image)

    ## save one HDF5 file for each zone
    for zone, images in zone_dict.items():
        if len(images) == 0:
            continue

        images_array = np.array(images, dtype=np.uint8)

        h5_path = os.path.join(subject_path, zone, f"{zone}.h5")

        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset(
                "images",
                data=images_array,
                compression="gzip",
                compression_opts=4         
            )
        

    
def read_subfolder(video_folder, output_folder, zones):
    """
    read and exract the frames for subject's subfolder. 
    If the subfolder contains a .zip file, extract the frames from the .zip file and save them in the output folder.
    Else, save the frames in the output folder
    """
    #chechk folder is empty
    if not os.listdir(video_folder):
        print(f"Warning: The folder {video_folder} is empty.")
    
    for i, sub_folder in enumerate(os.listdir(video_folder)):
        
        print(f"{i}) Processing subfolder: {sub_folder}")
        # find the .zip file in the subfolder
        zip_file = None
        for file in os.listdir(os.path.join(video_folder, sub_folder)):
            if file.endswith(".zip"):
                zip_file = file
                break
        print(f"Found zip file: {zip_file}")
        
        if zip_file is not None:
            zip_path = os.path.join(video_folder, sub_folder, zip_file)
            # extrapolate the frames from the .zip file and save them in the output folder
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(video_folder, sub_folder))

            create_and_fill_zone_folder(os.path.join(video_folder, sub_folder, 'images'), output_folder, sub_folder, zones)

            # remove the extracted folder
            shutil.rmtree(os.path.join(video_folder, sub_folder, 'images'))

        else:
            # create
            create_and_fill_zone_folder(os.path.join(video_folder, sub_folder), output_folder, sub_folder, zones)
            pass


def main(args):
    """
    Processing data and create the dataset
    """
    zones = ['z01', 'z02', 'z03', 'z04', 'z05', 'z06', 'z07', 'z08', 'z09', 'z10', 'z11', 'z12']

    # create output_folder if not exist
    os.makedirs(args.output_folder, exist_ok=True)

    ## read and exract the frames for sunjects's subfolder
    read_subfolder(args.videos_folder, args.output_folder, zones)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the LUS dataset from OpenPOCUS data")
    parser.add_argument("--metadata_file", type=str, help="The path to the metadata file", 
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/Database for Github.xlsx")
    parser.add_argument("--videos_folder", type=str, help="The path to the videos folder", 
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/Lung_Database/Lung_Database")
    parser.add_argument("--output_folder", type=str, help="The path to the output folder",
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/OpenPOCUS/DATA_Lung_Database")
    args = parser.parse_args()

    main(args)