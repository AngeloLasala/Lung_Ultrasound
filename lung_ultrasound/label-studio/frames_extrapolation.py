"""
Convert video into frames. This is crutial for getting the sutable number of frames and formation to imort in Label-studo for annotation
Note: use this code if the protocal deal with frame level segmentation
"""
import os 
import argparse
import cv2 
import tqdm
import json
import h5py

def main(args):
    """
    Create folder with extrapolated frames
    """
    ## create zones list
    zones = {'z1':'z01','z2':'z02','z3':'z03','z4':'z04','z5':'z05',
                    'z6':'z06','z7':'z07','z8':'z08','z9':'z09','z10':'z10','z11':'z11','z12':'z12'}
    ## Read data
    dataset_path = os.path.join(args.main_path, args.dataset)
    subjects_dict_path = os.path.join(args.main_path, args.dataset, args.splitting)
    subjects_dict = json.load(open(subjects_dict_path, 'r'))

    subjects_list = []
    for ss in ['train', 'val', 'test']:
        subjects_list += subjects_dict['fold_1'][ss]

    ## Create frames folder
    frames_folder = os.path.join(args.main_path, args.frames_path)
    os.makedirs(frames_folder, exist_ok=True)

    for sub in subjects_list:
        sub_path = os.path.join(frames_folder, sub)
        os.makedirs(sub_path, exist_ok=True)

    ## for each patient read video for each zones
    n = 0
    for sub in subjects_list:
        sub_videos_path = os.path.join(args.main_path, args.dataset, sub)

        labels_dict = os.path.join(sub_videos_path, 'labels.json')
        labels_dict = json.load(open(labels_dict, 'r'))
        
        for zone in zones.keys():
            if labels_dict[zone] != 'Nan':
                video_frames_path = os.path.join(sub_videos_path, zones[zone], f"{zones[zone]}.h5")

                with h5py.File(video_frames_path, "r") as f:
                    video_df = f["images"]
                    video_frames = video_df[:]

                sampling_frame = video_frames[::args.sampling_step]    
                n += sampling_frame.shape[0]

                for idx, frame in enumerate(sampling_frame):
                    filename = f"{sub}_{zone}_frame_{idx:04d}.png"
                    path = os.path.join(frames_folder, sub, filename)
                    cv2.imwrite(path, frame)

    print(f'Total frames: {n}')  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extrapolate frames from video')
    parser.add_argument('--main_path', help='Path to the main folder, i.e. ../OpenPOCUS')
    parser.add_argument('--dataset', help='Name of the dataset folder, i.e. DATA_Lung_Database')
    parser.add_argument('--splitting', help='Name of the splitting folder with the id of patients in dataset, i.e. splitting_10_percentage.json')
    parser.add_argument('--frames_path', help='Path to the output folder where frames will be saved, i.e. DATA_extrapolate_frames/')
    parser.add_argument('--sampling_step', default=10, type=int, help='sampling step, 10 = 30fps/3fps -> sampling at 3fps')
    # parser.add_argument('--time_window_per_folder', default=6, type=int, help='time windows (sec) of each subfolder')
    args = parser.parse_args()

    main(args)
    exit()

    ## dataset configuration
    size = (64,64)
    im_channels = 1
    fold_cv = 'fold_1'        # cross-validation fold 
    splitting = 'splitting.json'
    lenght = 1               # lenght of the segments of frames to return (in seconds)
    overlap = 0.2            # overlap between segments (percentage between 0 and 1)
    sampling_f = 30          # sampling frequency of the frames (default: 30)
    fps = 30

    # Create the output folder if it doesn't exist
    os.makedirs(args.frame_folder, exist_ok=True)

    for video_name in os.listdir(args.video_folder):
        video_path = os.path.join(args.video_folder, video_name)
        video_name_dict = video_name.split('.')[0]

        # Import Video from main folder: 'Data/'
        video = cv2.VideoCapture(video_path)

        # Extrapolate all the frames and give the main information on the video
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
        
        ## Downsampling the input video with given sampling frequency args.sampling_fps
        step = fps / args.sampling_fps  # updating 30 with variable fps
        frame_per_folder = round( (round(fps) / args.sampling_fps) * round(step) * args.time_window_per_folder)
        count = 0
        batch_index = 0


        print(f' fps:{round(fps)}, sampling_fps:{args.sampling_fps}, sampling step:{round(step)}, frame_per_folder:{frame_per_folder}, time window:{args.time_window_per_folder}')
        for frame_id in tqdm.tqdm(range(total_frames)):
            # Read the next frame
            ret, frame = video.read()

            if not ret or frame is None:
                break

            # Check if it's time to save this frame
            if (frame_id % round(step)) == 0:

                # Verifica se bisogna creare una nuova sottocartella
                if count % frame_per_folder == 0:
                    batch_folder = os.path.join(args.frame_folder, video_name_dict, f'batch_{batch_index:04d}', 'images')
                    os.makedirs(batch_folder, exist_ok=True)
                    batch_index += 1

                frame = cv2.resize(frame, args.resize)

                frame_filename = os.path.join(batch_folder, f"{count:05d}.jpg")
                cv2.imwrite(frame_filename, frame)

            count += 1
        video.release()