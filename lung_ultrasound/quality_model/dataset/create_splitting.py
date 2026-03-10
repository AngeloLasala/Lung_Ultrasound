"""
Create json 5 fold cross validation splitting at patient level
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
    subjects_list = [x for x in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, x))]
    
    np.random.seed(42)
    np.random.shuffle(subjects_list)
    # create a 5 fold cross validation
    n_folds = 5
    folds = np.array_split(subjects_list, n_folds)
    
    splitting_dict = {}
    for fold in range(n_folds):
        test_subjects = folds[fold].tolist()
        val_subjects = folds[(fold + 1) % n_folds].tolist()
        train_subjects = [subject for f in range(n_folds) if f != fold and f != (fold + 1) % n_folds for subject in folds[f].tolist()]
        
        splitting_dict[f'fold_{fold+1}'] = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }

    splitting_file_path = os.path.join(args.dataset_path, "splitting_1.json")
    with open(splitting_file_path, 'w') as f:
        json.dump(splitting_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the LUS dataset from OpenPOCUS data")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset")
                        
    args = parser.parse_args()

    main(args)