"""
Once the F fold cross validation provide you the best strategy, train the model with the selected best strategy
using THE ENTIRE DATASET for training and validation.

This code provide the 80:20 splitting for traing and val
"""
import argparse
import os
import numpy as np
from pathlib import Path
import json
import shutil
import zipfile
import h5py

def main(args):
    """
    Create a json file name splitting.json with the splitting of the dataset
    """
    # subjects_list = os.listdir(args.dataset_path)
    subjects_list = [x for x in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, x)) and args.exclude_sbj not in x.lower()]   # exclude liver subjects

    np.random.seed(42)
    np.random.shuffle(subjects_list)
    
    # define single fold split    
    splitting_dict = {'fold_inference':{'train':[], 'val':[]}}
    
    # 80:20 split
    split_idx = int(0.8 * len(subjects_list))
    train_subjects = subjects_list[:split_idx]
    val_subjects = subjects_list[split_idx:]
            
    splitting_dict['fold_inference'] = {
            'train': train_subjects,
            'val': val_subjects,
        }

    splitting_file_path = os.path.join(args.dataset_path, "splitting_inference.json")
    with open(splitting_file_path, 'w') as f:
        json.dump(splitting_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the LUS dataset from OpenPOCUS data")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset")
    parser.add_argument("--exclude_sbj", type=str, help="exclude liver or plax subject, (liver, plax)" )    
    args = parser.parse_args()

    main(args)