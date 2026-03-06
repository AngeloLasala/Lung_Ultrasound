"""
Helper function for fancy visualizing prediction
"""
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend non-interattivo per salvare figure
from PIL import Image
import io

def visualize_inference(video, pred):
    """
    Process video in batches of ff frames, collect overlays, return dict of PIL images.
    """
    F, C, H, W = pred.shape

    masks = torch.argmax(pred, dim=1).detach().cpu().numpy()  # F x H x W
    video_np = video.detach().cpu().numpy()[:F, 0, :, :]      # F x H x W

    frames_dict = {"frame": [], "frame_pleura": [], "frame_ribs": []}

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
    