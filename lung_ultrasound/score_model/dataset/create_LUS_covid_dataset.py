"""
Create LUS dataset form the data presented ONLY in official repository:
https://github.com/jannisborn/covid19_ultrasound/tree/master
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

def read_metadata(args):
    """
    Main function that take the argment and process based on user comand
    """

    # read the metadata file as a pandas datafreme
    metadata_df = pd.read_csv(args.metadata_file, encoding="latin1")

    columns_to_keep = [
        'Current location',
        'Filename',
        'URL (Video Name)',
        'Gender',
        'Age',
        'Patient ID / Name',
        'Label',
        'Lung Severity Score',
        'Type',
        'Probe',
        'Resolution',
        'Framerate',         
        'Length (frames)',
        'Effusion', 'Consolidations',
        'B-lines', 'A-lines', 'Pleural line irregularities', 'Air bronchogram'
    ]

    metadata_df = metadata_df[columns_to_keep]

    return metadata_df

def plot_video_frames(video_path, sampling_frequency=3,  output_folder=None, save_frames=True, show_plot=True):
    """
    Read video and extrapolate the frames
    
    Parameters:
    -----------
    video_path : str
        video path
    sampling_frequency : int
        sampling frequency in Hz
    """
    
    
    valid_extensions = {'.mp4', '.MP4', '.avi', '.mov', '.gif', '.mpeg'}
    cap = cv2.VideoCapture(video_path)
    
    # Video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

   
    # get frame index based on a sampling frequqenxies of 3HZ
    interval = max(1, int(round(fps / sampling_frequency)))  # <-- fix: interval must be >=1
    frame_indeces = list(range(0, total_frames, interval))
 
    # Estrai i frame
    frames = []
    for idx in frame_indeces:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Converti da BGR (OpenCV) a RGB (Matplotlib)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame_rgb)
    
    cap.release()
    
    # Plot dei frame
    if frames:
        for i, frame in enumerate(frames):
            if save_frames:
                frame_path = os.path.join(output_folder, f'frame_{frame_indeces[i]:04d}.png')
                cv2.imwrite(frame_path, frame)
    else:
        print("No frames extracted")
        
    # if frames:
    #     n_frames = len(frames)
    #     cols = min(5, n_frames)
    #     rows = (n_frames + cols - 1) // cols
        
    #     fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows), tight_layout=True)
    #     fig.suptitle(f'extrapolate frames: {Path(video_path).name}', fontsize=16)
        
    #     # Gestisci il caso di un singolo frame
    #     if n_frames == 1:
    #         axes = np.array([axes])
        
    #     axes = axes.flatten() if n_frames > 1 else axes
        

    #     for i, (ax, frame) in enumerate(zip(axes, frames)):
    #         ax.imshow(frame, cmap='gray')
    #         ax.set_title(f'Frame {frame_indeces[i]}')
    #         ax.axis('off')
        
    #         if save_frames:
    #             frame_path = os.path.join(output_folder, f'frame_{frame_indeces[i]:04d}.png')
    #             cv2.imwrite(frame_path, frame)

            
    #     # Nascondi gli assi non utilizzati
    #     for i in range(n_frames, len(axes)):
    #         axes[i].axis('off')
        
    #     plt.tight_layout()
    #     if show_plot : plt.show()
    # else:
    #     print("Not extrapoled frames")
    

def main(args):
    """
    Create the dataset 
    """
    # get metadata
    metadata_df = read_metadata(args)

    # create output folder if not exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # get list of viode filename
    type_of_probe = ['convex'] #, 'linear']
    video_paths = [os.path.join(args.videos_folder, probe, filename) for probe in type_of_probe for filename in os.listdir(os.path.join(args.videos_folder, probe))]

    for video_path in tqdm.tqdm(video_paths):
        # check if the file is a video:
        if not any(video_path.endswith(ext) for ext in ['.mp4', '.MP4', '.avi', '.mov', '.gif', '.mpeg']):
            print(f"File {os.path.basename(video_path)} is not a valid video file. Skipping.")
            continue

        # check if the pfs is greater than 29 fps
        if cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS) < 29:
            print(f"Video {video_path} has a frame rate of {cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)} fps, which is less than 29 fps. Skipping.")
            continue
        
        filename = os.path.basename(video_path)
        filename_without_extension = os.path.splitext(filename)[0]
        extention = os.path.splitext(filename)[1]
        
        if filename_without_extension not in metadata_df['Filename'].values:
            print(f"Filename {filename} not found in metadata")

        else:
            line = metadata_df[metadata_df['Filename'] == filename_without_extension]
            
            # creare a folder for each video
            video_output_folder = os.path.join(args.output_folder, filename_without_extension)
            os.makedirs(video_output_folder, exist_ok=True)
            video_images_folder = os.path.join(video_output_folder, "images")
            video_labels_folder = os.path.join(video_output_folder, "labels")
            os.makedirs(video_images_folder, exist_ok=True)
            os.makedirs(video_labels_folder, exist_ok=True)
            
            sammpling_frequency = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
            plot_video_frames(video_path, sampling_frequency=sammpling_frequency, output_folder=video_images_folder, save_frames=True, show_plot=False)

            # save label as json file
            label_info = line.to_dict(orient='records')[0]
            label_file_path = os.path.join(video_labels_folder, "label_info.json")
            with open(label_file_path, 'w') as f:
                json.dump(label_info, f, indent=4)

    ## create a json file name splitting.json with the splitting of the dataset
    subjects_list = os.listdir(args.output_folder)
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

    splitting_file_path = os.path.join(args.output_folder, "splitting.json")
    with open(splitting_file_path, 'w') as f:
        json.dump(splitting_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read the metadata info form file")
    parser.add_argument("--metadata_file", type=str, help="The path to the metadata file", 
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/covid19_ultrasound/data/dataset_metadata.csv")
    parser.add_argument("--videos_folder", type=str, help="The path to the videos folder", 
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/covid19_ultrasound/data/pocus_videos")
    parser.add_argument("--output_folder", type=str, help="The path to the output folder",
                        default="/media/angelo/PortableSSD/Assistant_Researcher/Predict/LUS_data_covid19/DATA_covid_compvital")
    args = parser.parse_args()

    main(args)
