"""
Create dataset from annotations in Label-Studio for the segmentation task od Rib shadow and Pleura
"""
import os 
import argparse
import cv2 
import tqdm
import json
import h5py


def main(args):
    """
    Read export file and return the dataset folder with images (olready have from frames_extrapolation) and labels
    """
    ## Read raw
    images_path = os.path.join(args.main_path, args.frames_path, args.subject, 'images')
    print(len(os.listdir(images_path)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extrapolate frames from video')
    parser.add_argument('--main_path', help='Path to the main folder, i.e. ../OpenPOCUS')
    parser.add_argument('--frames_path', help='Path to the output folder where frames will be saved, i.e. DATA_extrapolate_frames/')
    parser.add_argument('--subject', help='Identifier of the patient, i.e. pt149')
    args = parser.parse_args()

    main(args)