"""
Generate custom splits_final.json for nnUNet from subject-level splitting.

Usage:
    python create_nnunet_splitting.py \
        --dataset_path path/to/original/dataset \
        --splitting splitting.json \
        --nnunet_raw path/to/nnUNet_raw/DatasetXXX_NAME \
        --nnunet_preprocessed path/to/nnUNet_preprocessed/DatasetXXX_NAME
"""

import os
import json
import argparse


def subject_to_pattern(subject_name: str) -> str:
    """
    Convert subject name to the prefix used in saved filenames.

    Examples:
        Pt49.23  ->  Pt49_23   (dot becomes underscore)
        Pt115    ->  Pt115     (unchanged)
        pt185    ->  pt185     (unchanged)
        ED8      ->  ED8       (unchanged)
    """
    return subject_name.replace('.', '_')


def get_case_ids_for_subject(subject: str, all_case_ids: list[str]) -> list[str]:
    """
    Return all case_ids that belong to a given subject.
    Uses '_z' boundary to avoid partial matches:
        'Pt11_z' matches 'Pt11_z1_frame_0000' but NOT 'Pt115_z12_frame_0002'
    """
    prefix = subject_to_pattern(subject) + '_z'
    return [c for c in all_case_ids if c.startswith(prefix)]


def load_case_ids_from_imagesTr(nnunet_raw: str) -> list[str]:
    """
    Scan imagesTr folder and extract case IDs (strip _0000.png suffix).
    E.g. 'Pt115_z12_frame_0002_0000.png' -> 'Pt115_z12_frame_0002'
    """
    imagesTr_path = os.path.join(nnunet_raw, 'imagesTr')
    case_ids = []
    for fname in sorted(os.listdir(imagesTr_path)):
        if fname.endswith('_0000.png'):
            case_ids.append(fname.replace('_0000.png', ''))
    return case_ids


def main(args):
    # --- Load subject-level splitting from original dataset ---
    with open(os.path.join(args.dataset_path, args.splitting)) as f:
        subject_splitting = json.load(f)

    # --- Load all case IDs from nnUNet_raw/DatasetXXX/imagesTr ---
    all_case_ids = load_case_ids_from_imagesTr(args.nnunet_raw)
    print(f"Found {len(all_case_ids)} total cases in imagesTr\n")

    # --- Build nnUNet splits_final.json ---
    # Format: list of dicts [{'train': [...], 'val': [...]}, ...]
    # Test subjects are excluded from both train and val
    splits_final = []

    for fold_key in sorted(subject_splitting.keys()):  # fold_1, fold_2, ...
        fold = subject_splitting[fold_key]

        fold_train_cases = []
        fold_val_cases = []

        for subject in fold['train']:
            cases = get_case_ids_for_subject(subject, all_case_ids)
            if not cases:
                print(f"  [WARNING] {fold_key} - no cases found for train subject '{subject}'")
            fold_train_cases.extend(cases)

        for subject in fold['val']:
            cases = get_case_ids_for_subject(subject, all_case_ids)
            if not cases:
                print(f"  [WARNING] {fold_key} - no cases found for val subject '{subject}'")
            fold_val_cases.extend(cases)

        splits_final.append({
            'train': sorted(fold_train_cases),
            'val':   sorted(fold_val_cases)
        })

        print(f"{fold_key}: {len(fold_train_cases):4d} train cases | "
              f"{len(fold_val_cases):4d} val cases | "
              f"{len(fold['test'])} test subjects excluded")

    # --- Sanity check: no overlap between train and val ---
    print()
    for i, fold in enumerate(splits_final):
        overlap = set(fold['train']) & set(fold['val'])
        if overlap:
            print(f"  [ERROR] fold_{i+1} has {len(overlap)} overlapping cases between train and val!")
        else:
            print(f"  fold_{i+1} OK - no train/val overlap")

    # --- Save splits_final.json into nnUNet_preprocessed/DatasetXXX ---
    os.makedirs(args.nnunet_preprocessed, exist_ok=True)
    output_file = os.path.join(args.nnunet_preprocessed, 'splits_final.json')
    with open(output_file, 'w') as f:
        json.dump(splits_final, f, indent=2)

    print(f"\nSaved -> {output_file}")
    print(f"Total folds: {len(splits_final)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create nnUNet custom splits_final.json')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the original dataset (where splitting.json lives)')
    parser.add_argument('--splitting', type=str, required=True,
                        help='Name of the splitting file, e.g. splitting.json')
    parser.add_argument('--nnunet_raw', type=str, required=True,
                        help='Path to nnUNet_raw/DatasetXXX_NAME (parent of imagesTr)')
    parser.add_argument('--nnunet_preprocessed', type=str, required=True,
                        help='Path to nnUNet_preprocessed/DatasetXXX_NAME (where splits_final.json will be saved)')
    args = parser.parse_args()
    main(args)