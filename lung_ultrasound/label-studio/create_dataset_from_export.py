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
    Read export file and return the dataset folder with images (already have from frames_extrapolation) and labels
    """
    ## Read raw
    subjects_path = os.path.join(args.main_path, args.frames_path)
    subjects_list = os.listdir(subjects_path)

    ## Check valid Subject
    print('Checking valid subject...')
    subjects_valid = []
    for subject in subjects_list:
        subject_path = os.path.join(subjects_path, subject)

        # controlla che sia una directory
        if not os.path.isdir(subject_path):
            continue

        images_path = os.path.join(subject_path, "images")

        # cerca un file json nella cartella del subject
        json_files = [f for f in os.listdir(subject_path) if f.endswith(".json")]

        if not os.path.exists(images_path):
            print(f"[WARNING] {subject} has no 'images' folder")
            continue

        if len(json_files) == 0:
            print(f"[WARNING] {subject} has no JSON annotation file")
            continue

        if len(json_files) > 1:
            print(f"[WARNING] {subject} has multiple JSON files, taking the first one")

        annotation_path = os.path.join(subject_path, json_files[0])

        subjects_valid.append({
            "name": subject,
            "images_path": images_path,
            "annotation_path": annotation_path
        })

    print(f"Found {len(subjects_valid)} valid subjects")
    print()

    print('Extrapolate annotations fro valid subject...')
    for subject in subjects_valid:
        name = subject['name']
        images_path = subject['images_path']
        annotation_path = subject['annotation_path']
        print(name)

        # read annotation path with json
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        ## check if the numer of annotations is equals to the number of images
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        images_list = [f for f in os.listdir(images_path) if f.lower().endswith(valid_ext)]
        n_images = len(images_list)
        n_annotations = len(annotations)

        if n_images != n_annotations:
            raise ValueError(
                f"Mismatch in subject '{name}': "
                f"{n_images} images found but {n_annotations} annotations found."
            )

        # read annotation
        for ann in annotations:
            print(ann.keys())
            image_name = ann['image'].split('-')[-1]
            segmentation_rle = ann['tag']

            original_w = segmentation_rle['original_width']
            original_h = segmentation_rle['original_height']
            print(segmentation_rle[0].keys(), original_w, original_h)

            plue
        exit()
        print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extrapolate frames from video')
    parser.add_argument('--main_path', help='Path to the main folder, i.e. ../OpenPOCUS')
    parser.add_argument('--frames_path', help='Path to the output folder where frames will be saved, i.e. DATA_extrapolate_frames/')
    args = parser.parse_args()

    main(args)