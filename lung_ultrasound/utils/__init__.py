"""
Utils files
"""
import os
import torch
import numpy as np
from PIL import Image

def get_frames_from_video(video_path, lenght, overlap, size,
                          fps=30, sampling_f=30):
    """
    From a folder with frames, retarn a list of segments of frames with the specified length and overlap.

    Parameters
    ----------
    video_path: str
        Path to the folder with the frames of the video. The frames should be named as "frame_00001.png", "frame_00002.png", etc.
    lenght: int
        Length of the segments of frames to return (in seconds)
    overlap: float
        Overlap between segments (percentage between 0 and 1)
    size: tuple
        Size of the frames to return (width, height)
    fps: int
        Frames per second of the video (default: 30)
    sampling_f: int
        Sampling frequency of the frames (default: 30)
    """

    total_video_frames = len(os.listdir(video_path))
    total_duration = total_video_frames / fps
    segment_length_frames = int(lenght * sampling_f)
    step_frames = int(segment_length_frames * (1 - overlap))

    segments = []
    for start in range(0, total_video_frames - segment_length_frames + 1, step_frames):
        end = start + segment_length_frames
        segments.append((start, end))

    frames = []
    for start, end in segments:
        segment_frames = []
        for frame_idx in range(start, end):
            frame_path = os.path.join(video_path, f"frame_{frame_idx:04d}.png")
            if os.path.exists(frame_path):
                frame = Image.open(frame_path)
                frame = frame.resize((size))
                frame = (np.array(frame))#/ 255.0
                segment_frames.append(frame)
            else:
                print(f"Warning: Frame {frame_path} does not exist.")
                # segment_frames.append(torch.zeros((1, 64, 64)))  # Placeholder for missing frames
        segment_frames = np.stack(segment_frames, axis=0)
        frames.append(segment_frames)

    return frames


def confusion_matrix(labels, preds, classes):
    """
    Compute the confusion matrix.

    Parameters
    ----------
    labels: list
        List of ground truth labels.
    preds: list
        List of predicted labels.
    classes: list
        List of class names.
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    import matplotlib.pyplot as plt

    num_classes = len(classes)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    cm = confusion_matrix(labels, preds)

    ## print classification report
    print("Classification Report:")
    print(classification_report(labels, preds, target_names=classes))
    report = classification_report(labels, preds, target_names=classes)

    ## compute the chance level for each class and the overall chance level
    total_samples = len(labels)
    class_counts = np.array([(np.array(labels) == c).sum() for c in range(num_classes)])

    class_probs = class_counts / total_samples
    print("\nChance Level Per Class (Class Prevalence):")
    for cls, prob in zip(classes, class_probs):
        print(f"{cls}: {prob:.4f} ({prob*100:.2f}%)")
        
    stratified_chance = np.sum(class_probs ** 2)  # Stratified random guessing accuracy
    majority_chance = np.max(class_probs) # Majority class baseline
    print("\nBaseline Chance Levels:")
    print(f"Stratified random chance accuracy: {stratified_chance:.4f} ({stratified_chance*100:.2f}%)")
    print(f"Majority-class baseline accuracy: {majority_chance:.4f} ({majority_chance*100:.2f}%)\n")
    

    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_sum[cm_sum == 0] = 1
    cm_norm = cm.astype(float) / cm_sum * 100

    ## plot cm
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im0 = axes[0].imshow(cm, cmap="Blues")

    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    axes[0].set_xticks(np.arange(num_classes))
    axes[0].set_yticks(np.arange(num_classes))
    axes[0].set_xticklabels(classes, rotation=45, ha="right")
    axes[0].set_yticklabels(classes)

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            axes[0].text(
                j, i,
                f"{value}",
                ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black"
            )

    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Number of samples")


    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)

    axes[1].set_title("Confusion Matrix (%)")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")

    axes[1].set_xticks(np.arange(num_classes))
    axes[1].set_yticks(np.arange(num_classes))
    axes[1].set_xticklabels(classes, rotation=45, ha="right")
    axes[1].set_yticklabels(classes)

    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_norm[i, j]
            axes[1].text(
                j, i,
                f"{value:.1f}%",
                ha="center", va="center",
                color="white" if value > 50 else "black"
            )

    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("Percentage (%)")


    plt.tight_layout()
    
    return report, fig, stratified_chance, majority_chance
            