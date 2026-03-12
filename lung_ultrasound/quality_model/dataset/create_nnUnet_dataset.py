"""
From the custom dataset folder, create the folder structure to train nnUnet model 
for segmentation of plax and ribs' shadow.

Note - Use the venv of nnUnet repository: https://github.com/MIC-DKFZ/nnUNet
"""
import os
import json
import argparse
import re

import numpy as np
from PIL import Image
import tqdm
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def build_case_id(name):
    """
    Convert: Pt49_23_z9_frame_0001 -> p4923z9f0001
    """
    m = re.match(r'Pt(\d+)_(\d+)_z(\d+)_frame_(\d+)', name)
    if m:
        a, b, z, f = m.groups()
        return f"p{a}{b}z{z}f{f}"
    return name.replace(".", "_")


def main(args):
    """
    Convert existing dataset folder for nnUnet training.
    All splits (train+val+test) go into imagesTr for custom 5-fold CV.
    """
    dataset_path = args.dataset_path
    main_path = os.path.dirname(dataset_path)

    # --- Create nnUNet_raw folder ---
    nnUnet_raw = os.path.join(main_path, 'nnUNet_raw')
    os.makedirs(nnUnet_raw, exist_ok=True)

    # --- Dataset name with incremental counter ---
    count = len(os.listdir(nnUnet_raw)) + 1
    dataset_name = f"Dataset{count:03d}_{args.name_dataset}"
    dataset_out_path = os.path.join(nnUnet_raw, dataset_name)
    os.makedirs(dataset_out_path, exist_ok=True)

    # --- ALL images go into imagesTr (nnUNet custom split) ---
    train_nn_path = os.path.join(dataset_out_path, 'imagesTr')
    labels_nn_path = os.path.join(dataset_out_path, 'labelsTr')
    os.makedirs(train_nn_path, exist_ok=True)
    os.makedirs(labels_nn_path, exist_ok=True)

    # --- Load splitting JSON ---
    with open(os.path.join(args.dataset_path, args.splitting)) as file:
        splitting = json.load(file)

    # Use fold_1 to get the full subject list (train + val + test)
    fold_1 = splitting['fold_1']
    subjects = fold_1['train'] + fold_1['val'] + fold_1['test']

    n_train = 0  # will count total images saved

    for sbj in tqdm.tqdm(subjects):
        sbj_path = os.path.join(args.dataset_path, sbj)
        images_path = os.path.join(sbj_path, 'images')
        labels_path = os.path.join(sbj_path, 'labels')

        image_files = sorted(os.listdir(images_path))
        label_files = sorted(os.listdir(labels_path))

        for img_name, lab_name in zip(image_files, label_files):
            img_file = os.path.join(images_path, img_name)
            lab_file = os.path.join(labels_path, lab_name)

            base_name = os.path.splitext(img_name)[0]
            case_id = build_case_id(base_name)

            # ---- LOAD ----
            image = Image.open(img_file).convert('L')
            mask_np = np.load(lab_file)
            mask = Image.fromarray(mask_np.astype(np.uint8))

            # ---- RESIZE ----
            image = image.resize((256, 256), Image.BILINEAR)
            mask = mask.resize((256, 256), Image.NEAREST)

            # ---- SAVE IMAGE (suffix _0000 = channel 0) ----
            new_img_name = f"{case_id}_0000.png"
            image.save(os.path.join(train_nn_path, new_img_name))

            # ---- SAVE LABEL ----
            new_lab_name = f"{case_id}.png"
            mask.save(os.path.join(labels_nn_path, new_lab_name))

            n_train += 1

    print(f"\nTotal cases saved (n_train): {n_train}")

    # --- Generate dataset.json ---
    generate_dataset_json(
        output_folder=dataset_out_path,
        channel_names={0: "L"},
        labels={"background": 0, "pleura": 1, "ribs": 2},
        num_training_cases=n_train,
        file_ending=".png",
        dataset_name=dataset_name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create nnUNet dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset')
    parser.add_argument('--splitting', type=str, required=True,
                        help='Name of the splitting file, e.g. splitting.json')
    parser.add_argument('--name_dataset', type=str, required=True,
                        help='Name suffix for the dataset, e.g. PlaxRibs')
    args = parser.parse_args()
    main(args)