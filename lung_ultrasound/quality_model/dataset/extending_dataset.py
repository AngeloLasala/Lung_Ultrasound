"""
Extending lung dataset with other standard plane acquitions of other region of interest:
- Liver
- Heart (PLAX) 

The code create 5 fake ptient for liver a heart view to be added:
- dataset path: adding the folder with images and lable (null)
- 5CV: adding the name to splitting.json
"""
import argparse
import os
import numpy as np
from pathlib import Path
import json
import shutil
import random
from PIL import Image

def distribute_images_to_subjects(dataset_ext, n_subjects=5, n_images_per_subject=30):
    """
    Shuffles all available images once and splits them into n_subjects
    non-overlapping chunks of n_images_per_subject images each.
    """
    ext_category_path = dataset_ext
    all_images = [
        f for f in os.listdir(ext_category_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    total_needed = n_subjects * n_images_per_subject
    assert len(all_images) >= total_needed, (
        f"[ERROR] Not enough images in '{ext_category_path}': "
        f"need {total_needed} ({n_subjects} subjects x {n_images_per_subject} imgs), "
        f"found {len(all_images)}"
    )

    # Shuffle once, then slice into non-overlapping chunks
    random.shuffle(all_images)
    chunks = [
        all_images[i * n_images_per_subject : (i + 1) * n_images_per_subject]
        for i in range(n_subjects)
    ]
    return ext_category_path, chunks


def create_fake_subject(dataset_path, ext_category_path, subject_name, selected_images):
    """
    Creates a fake subject folder structure:
    dataset_path/
    └── subject_name/
        ├── images/   <- pre-assigned non-overlapping image chunk, saved as grayscale .png
        └── labels/   <- one .npy file per image, all zeros, shape (H, W)
    """
    images_folder = os.path.join(dataset_path, subject_name, "images")
    labels_folder = os.path.join(dataset_path, subject_name, "labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    for img_name in selected_images:
        src = os.path.join(ext_category_path, img_name)

        # Force grayscale and save as .png regardless of original format
        img = Image.open(src).convert("L")
        img_name_png = os.path.splitext(img_name)[0] + ".png"
        dst = os.path.join(images_folder, img_name_png)
        img.save(dst)

        # Zero label with same shape as grayscale image -> (H, W), no channel dim
        img_array = np.array(img)
        zero_label = np.zeros(img_array.shape, dtype=img_array.dtype)
        label_name = os.path.splitext(img_name)[0] + ".npy"
        np.save(os.path.join(labels_folder, label_name), zero_label)

    print(f"[OK] '{subject_name}' <- {len(selected_images)} images saved as grayscale .png (no overlaps guaranteed)")


def add_subject_to_splits(splitting, subject_name, assigned_fold):
    """
    Adds subject_name to the splitting dict following standard 5-fold CV logic:
      - test  in assigned_fold
      - val   in the fold before assigned_fold
      - train in the remaining 3 folds
    """
    folds = sorted(splitting.keys())  # ['fold_1', ..., 'fold_5']
    n_folds = len(folds)

    assigned_idx = folds.index(assigned_fold)
    val_idx = (assigned_idx - 1) % n_folds  # fold before is val

    for i, fold_name in enumerate(folds):
        if i == assigned_idx:
            splitting[fold_name]["test"].append(subject_name)
        elif i == val_idx:
            splitting[fold_name]["val"].append(subject_name)
        else:
            splitting[fold_name]["train"].append(subject_name)

    return splitting


def main(args):
    dataset_path = args.dataset_path
    dataset_ext = args.dataset_ext
    n_subjects = 5      # one per fold
    n_images   = 30     # images per subject

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Read original splitting.json
    json_path = os.path.join(dataset_path, f'{args.original_json}.json')
    with open(json_path, 'r') as file:
        splitting = json.load(file)

    folds = sorted(splitting.keys())  # ['fold_1', 'fold_2', ..., 'fold_5']
    assert len(folds) == n_subjects, f"Expected 5 folds, found {len(folds)}"

    # for category in categories:
    category = os.path.basename(dataset_ext).split('_')[-1]


    # Distribute images ONCE across all 5 subjects — no overlaps
    ext_category_path, image_chunks = distribute_images_to_subjects(
        dataset_ext=dataset_ext,
        n_subjects=n_subjects,
        n_images_per_subject=n_images
    )

    for i, (fold_name, image_chunk) in enumerate(zip(folds, image_chunks)):
        subject_name = f"Pt_{category}_{i + 1}"  # e.g. Pt_liver_1

        # 1. Create fake subject folder with its exclusive image chunk
        create_fake_subject(
            dataset_path=dataset_path,
            ext_category_path=ext_category_path,
            subject_name=subject_name,
            selected_images=image_chunk
        )

        # 2. Add to splitting: test in assigned fold, val in previous, train in rest
        splitting = add_subject_to_splits(splitting, subject_name, fold_name)
        print(f"  -> '{subject_name}': test='{fold_name}', "
                f"val='{folds[(i - 1) % n_subjects]}', train=other 3 folds")

    # Save updated splitting.json (new file, original untouched)
    output_json_path = os.path.join(dataset_path, f'{args.original_json}_ext_{category}.json')
    with open(output_json_path, 'w') as file:
        json.dump(splitting, file, indent=4)

    print(f"\n[DONE] Updated splitting saved to: {output_json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the LUS dataset from OpenPOCUS data")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset")
    parser.add_argument("--dataset_ext", type=str, help="The path to the extended dataset")
    # parser.add_argument('--frames_path', help='Path to the output folder where frames will be saved, i.e. Extrapolate_frames/')
    parser.add_argument("--original_json", type=str, help="original splitting.json file, fake subject are added in this splitting")

    args = parser.parse_args()

    main(args)
