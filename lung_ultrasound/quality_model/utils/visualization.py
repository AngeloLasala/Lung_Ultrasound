"""
Helper function for fancy visualizing prediction
"""
import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg')  # backend non-interattivo per salvare figure
from PIL import Image
import io

def compute_centroid(binary_mask):
    """
    Compute centroids from binary bask
    """
    coords = np.argwhere(binary_mask)  # shape (N, 2) -> (row, col)
    if len(coords) == 0:
        return None
    cy, cx = coords.mean(axis=0)
    return (float(cx), float(cy))

def split_centroids(centroids):
    """
    Splic x and y centroids' coordinates
    """
    cx = np.array([c[0] if c is not None else np.nan for c in centroids])
    cy = np.array([c[1] if c is not None else np.nan for c in centroids])
    return cx, cy

def visualize_inference(video, pred):
    """
    Process video in batches of ff frames, collect overlays, return dict of PIL images.
    """
    F, C, H, W = pred.shape

    masks = torch.argmax(pred, dim=1).detach().cpu().numpy()  # F x H x W
    video_np = video.detach().cpu().numpy()[:F, 0, :, :]      # F x H x W

    frames_dict = {
        "frame": [],
        "frame_pleura": [],
        "frame_ribs": [],
        "centroid_pleura": [],   # list of (cx, cy) or None if class absent
        "centroid_ribs": [],     # list of (cx, cy) or None if class absent
    }

    for f in range(F):
        frame = video_np[f]
        mask  = masks[f]

        # Normalizza in uint8
        if frame.max() <= 1.0:
            frame_uint8 = (frame * 255).astype(np.uint8)
        else:
            frame_uint8 = frame.astype(np.uint8)

        frame_rgb = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2RGB)

        binary_mask1 = (mask == 1)  # pleura
        binary_mask2 = (mask == 2)  # ribs

        # Centroids
        frames_dict["centroid_pleura"].append(compute_centroid(binary_mask1))
        frames_dict["centroid_ribs"].append(compute_centroid(binary_mask2))

        # Overlays
        overlay1 = frame_rgb.copy()
        overlay1[binary_mask1] = [255, 80, 80]
        overlay1 = cv2.addWeighted(frame_rgb, 0.4, overlay1, 0.3, 0)  

        overlay2 = frame_rgb.copy()
        overlay2[binary_mask2] = [80, 180, 255]
        overlay2 = cv2.addWeighted(frame_rgb, 0.4, overlay2, 0.3, 0) 
        

        frames_dict["frame"].append(Image.fromarray(frame_rgb))
        frames_dict["frame_pleura"].append(Image.fromarray(overlay1))
        frames_dict["frame_ribs"].append(Image.fromarray(overlay2))
        
    return frames_dict

def make_gif(all_frames_dict, output_path="inference.gif", fps=10):
    """
    Crea una GIF con 3 colonne: frame | pleura | ribs
    """
    n_frames = len(all_frames_dict["frame"])
    duration_ms = int(1000 / fps)

    gif_frames = []

    for f in range(n_frames):
        img_frame   = np.array(all_frames_dict["frame"][f])
        img_pleura  = np.array(all_frames_dict["frame_pleura"][f])
        img_ribs    = np.array(all_frames_dict["frame_ribs"][f])

        # Affianca le 3 immagini orizzontalmente
        combined = np.concatenate([img_frame, img_pleura, img_ribs], axis=1)
        gif_frames.append(Image.fromarray(combined))

    # Salva GIF
    gif_frames[0].save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        loop=0,
        duration=duration_ms
    )
    print(f"GIF salvata in: {output_path}  ({n_frames} frames @ {fps} fps)")
    
def plot_centroids_over_time(frames_dict, fps=1, img_size=None, save_path=None, filename=None):
    """
    Plot temporal evolution of pleura and ribs centroids (cx and cy separately).
    
    Args:
        frames_dict: output of visualize_inference()
        fps:         frames per second, used to convert frame index to seconds
        img_size:    int or tuple (H, W), used to set y-axis limits. If None, limits are automatic.
        save_path:   folder where to save the plot (optional)
        filename:    filename for the saved plot, e.g. "centroids.png" (optional)
                     if save_path is given but filename is not, defaults to "centroids_plot.png"
    
    Returns:
        fig: matplotlib Figure
    """
    centroids_pleura = frames_dict["centroid_pleura"]
    centroids_ribs   = frames_dict["centroid_ribs"]
    F = len(centroids_pleura)
    time_axis = np.arange(F) / fps

    cx_pleura, cy_pleura = split_centroids(centroids_pleura)
    cx_ribs,   cy_ribs   = split_centroids(centroids_ribs)

    # derive axis limits from img_size
    if img_size is not None:
        if isinstance(img_size, (tuple, list)):
            h, w = img_size[0], img_size[1]
        else:
            h = w = img_size  # square image
        xlim = (0, w)
        ylim = (0, h)
    else:
        xlim = ylim = None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax = axes[0]
    ax.plot(time_axis, cx_pleura, color="tomato",         marker="o", markersize=5, linewidth=2, label="Pleura cx")
    ax.plot(time_axis, cx_ribs,   color="cornflowerblue", marker="o", markersize=5, linewidth=2, label="Ribs cx")
    ax.set_ylabel("cx  [px]", fontsize=13)
    ax.set_title("Centroid X position over time", fontsize=15)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    if xlim is not None:
        ax.set_ylim(*xlim)
    ax.invert_yaxis()  # pixel (0,0) is top-left, plot y increases downward

    ax = axes[1]
    ax.plot(time_axis, cy_pleura, color="tomato",         marker="o", markersize=5, linewidth=2, label="Pleura cy")
    ax.plot(time_axis, cy_ribs,   color="cornflowerblue", marker="o", markersize=5, linewidth=2, label="Ribs cy")
    ax.set_ylabel("cy  [px]", fontsize=13)
    ax.set_title("Centroid Y position over time", fontsize=15)
    ax.legend(fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Time [s]" if fps != 1 else "Frame", fontsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.invert_yaxis()  # pixel (0,0) is top-left, plot y increases downward

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fname = filename if filename is not None else "centroids_plot.png"
        full_path = os.path.join(save_path, fname)
        fig.savefig(full_path, dpi=150, bbox_inches="tight")
    return fig

def visualize_image_mask(image, label, num, save_path=None):
    """
    Plot image | image + overlay (pleura + ribs).
    
    Args:
        image:      torch.Tensor C x H x W or numpy H x W x C
        label:      torch.Tensor or numpy H x W, values 0/1/2
        idx:        index label for the plot title
        save_path:  optional folder where to save the figure
    """
    # Converti immagine
    if hasattr(image, 'permute'):
        image = image.permute(1, 2, 0).numpy()

    if image.max() <= 1.0:
        frame_uint8 = (image * 255).astype(np.uint8)
    else:
        frame_uint8 = image.astype(np.uint8)

    if frame_uint8.shape[2] == 1:
        frame_rgb = cv2.cvtColor(frame_uint8[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        frame_rgb = frame_uint8

    # Maschere
    label_np = label.numpy() if hasattr(label, 'numpy') else np.array(label)
    binary_mask1 = (label_np == 1)  # pleura
    binary_mask2 = (label_np == 2)  # ribs

    # Overlay
    overlay = frame_rgb.copy()
    overlay[binary_mask1] = [255, 80,  80 ]
    overlay[binary_mask2] = [80,  180, 255]
    overlay = cv2.addWeighted(frame_rgb, 0.4, overlay, 0.4, 0)

    # Plot
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), num=num)

    axes[0].imshow(frame_rgb, cmap='gray')
    # axes[0].set_title(f"{num}", fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    # axes[1].set_title("Image + Pleura/Ribs mask", fontsize=13)
    axes[1].axis('off')
    axes[1].legend(handles=[
        Patch(facecolor=[1, 0.31, 0.31], label='Pleura'),
        Patch(facecolor=[0.31, 0.71, 1],  label="Ribs' shadow"),
    ], loc='lower right', fontsize=13)

    plt.tight_layout()

    if save_path is not None:
        # os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f"{num}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()