"""
Inference script compatibile con nnUNet v2 (2D, grayscale).
Esegue manualmente il preprocessing nnUNet (legge plans.json e dataset.json)
e fa inferenza frame-per-frame senza usare nnUNetPredictor.
"""

import os
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import h5py
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from tqdm import tqdm

# nnUNet v2 imports
from nnunetv2.architecture.residual_encoders import ResidualEncoderUNet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

from lung_ultrasound.quality_model.utils.visualization import visualize_inference, make_gif, plot_centroids_over_time


# ---------------------------------------------------------------------------
# Helpers: load plans/dataset and build preprocessing params
# ---------------------------------------------------------------------------

def load_nnunet_config(plans_path: str, dataset_path: str, configuration: str = "2d"):
    """
    Legge plans.json e dataset.json e restituisce:
      - patch_size      : (H, W)
      - target_spacing  : (sp_y, sp_x)   [mm, può essere None se non disponibile]
      - norm_mean       : float
      - norm_std        : float
      - num_classes     : int  (incluso background)
      - network_config  : dict con architettura
    """
    with open(plans_path, "r") as f:
        plans = json.load(f)

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # --- patch size e spacing dal plans ---
    cfg_block = plans["configurations"][configuration]
    patch_size = tuple(cfg_block["patch_size"])          # es. [512, 512]
    target_spacing = cfg_block.get("spacing", None)      # es. [1.0, 1.0]

    # --- numero classi ---
    num_classes = len(dataset["labels"])                  # include background (label 0)

    # --- normalizzazione: nnUNet salva mean/std per canale in dataset.json ---
    # struttura attesa: dataset["channel_names"] + stats in plans o dataset
    # In nnUNet v2 i foreground intensity properties sono in plans
    intensity_props = plans.get("foreground_intensity_properties_per_channel", None)
    if intensity_props is not None:
        # chiave "0" per canale 0 (grayscale)
        ch0 = intensity_props.get("0", intensity_props.get(0, {}))
        norm_mean = float(ch0.get("mean", 0.0))
        norm_std  = float(ch0.get("std",  1.0))
    else:
        # fallback: nessuna normalizzazione
        print("[WARNING] foreground_intensity_properties_per_channel non trovato in plans.json — uso mean=0, std=1")
        norm_mean = 0.0
        norm_std  = 1.0

    return {
        "patch_size":     patch_size,
        "target_spacing": target_spacing,
        "norm_mean":      norm_mean,
        "norm_std":       norm_std,
        "num_classes":    num_classes,
        "configuration":  cfg_block,
        "plans":          plans,
    }


def build_model_from_plans(plans: dict, configuration: str, num_classes: int, device: torch.device):
    """
    Ricostruisce la rete nnUNet v2 a partire dal plans.json.
    Usa PlansManager / ConfigurationManager di nnunetv2.
    """
    plans_manager    = PlansManager(plans)
    config_manager   = plans_manager.get_configuration(configuration)
    num_input_channels = 1  # grayscale

    network = config_manager.network_arch_class_name  # es. 'PlainConvUNet'

    # nnUNet v2: build network tramite ConfigurationManager
    model = config_manager.get_network_architecture(
        num_input_channels=num_input_channels,
        num_classes=num_classes,
        enable_deep_supervision=False,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Preprocessing: replica esatta del nnUNet 2D preprocessor
# ---------------------------------------------------------------------------

def preprocess_frame(frame: np.ndarray,
                     patch_size: tuple,
                     norm_mean: float,
                     norm_std: float) -> torch.Tensor:
    """
    Preprocessing di un singolo frame grezzo (H, W) uint8 → tensore (1, 1, pH, pW).

    Passi replicati da nnUNet v2 DefaultPreprocessor (2D):
      1. Converti in float32
      2. Clipping ai percentili 0.5–99.5 (CT-style, ma applicato anche a US)
      3. Z-score normalisation con mean/std del foreground del training set
      4. Resize bilineare alla patch_size del piano
    """
    img = frame.astype(np.float32)

    # 1. Clipping ai percentili (nnUNet lo fa per CT; per US è comunque utile)
    p005 = np.percentile(img, 0.5)
    p995 = np.percentile(img, 99.5)
    img  = np.clip(img, p005, p995)

    # 2. Z-score con statistiche del training set
    if norm_std > 0:
        img = (img - norm_mean) / norm_std
    else:
        img = img - norm_mean

    # 3. Resize alla patch size con interpolazione bilineare
    ph, pw = patch_size
    img_resized = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LINEAR)

    # 4. Aggiungi dimensioni batch e canale → (1, 1, H, W)
    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


# ---------------------------------------------------------------------------
# Dataset paziente (invariato nella struttura, aggiornato nel preprocessing)
# ---------------------------------------------------------------------------

class PatientDataset(Dataset):
    """
    Carica tutti i video (h5 / mp4 / avi) di un paziente.
    Restituisce frame grezzi (numpy, uint8) senza preprocessing,
    che verrà applicato al momento dell'inferenza.
    """

    ZONES_MAP = {f"z{i}": f"z{i:02d}" for i in range(1, 13)}

    def __init__(self, subject_path: str, default_fps: int = 30):
        self.subject_path = subject_path
        self.default_fps  = default_fps
        self.fps          = default_fps

        data = self._load_all_zones()
        self.videos = data["videos"]   # list of np.ndarray (F, H, W)
        self.labels = data["labels"]   # list of str
        self.zones  = data["zones"]    # list of str  e.g. "z1"

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx], self.zones[idx]

    # ------------------------------------------------------------------
    def _load_all_zones(self):
        result = {"videos": [], "labels": [], "zones": []}

        labels_path = os.path.join(self.subject_path, "labels.json")
        with open(labels_path, "r") as f:
            subject_labels = json.load(f)

        for zone_key, zone_dir in self.ZONES_MAP.items():
            if subject_labels.get(zone_key, "Nan") == "Nan":
                continue

            zone_path = os.path.join(self.subject_path, zone_dir)
            h5_path   = os.path.join(zone_path, f"{zone_dir}.h5")
            mp4_path  = os.path.join(zone_path, f"{zone_dir}.mp4")
            avi_path  = os.path.join(zone_path, f"{zone_dir}.avi")

            if os.path.exists(h5_path):
                frames = self._load_h5(h5_path)
            elif os.path.exists(mp4_path):
                frames = self._load_video(mp4_path)
            elif os.path.exists(avi_path):
                frames = self._load_video(avi_path)
            else:
                print(f"[WARNING] Nessun file video trovato per zona '{zone_key}' — skip.")
                continue

            result["videos"].append(frames)
            result["labels"].append(subject_labels[zone_key])
            result["zones"].append(zone_key)

        return result

    def _load_h5(self, path: str) -> np.ndarray:
        with h5py.File(path, "r") as f:
            return f["images"][:]  # (F, H, W)

    def _load_video(self, path: str) -> np.ndarray:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or self.default_fps
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames:
            raise ValueError(f"No frames in: {path}")
        return np.stack(frames, axis=0)  # (F, H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ------------------------------------------------------------------
    # 1. Leggi plans.json e dataset.json → parametri preprocessing + architettura
    # ------------------------------------------------------------------
    print("\n[1/4] Caricamento configurazione nnUNet...")
    nnunet_cfg = load_nnunet_config(
        plans_path   = args.plans_path,
        dataset_path = args.dataset_json_path,
        configuration = args.configuration,
    )

    patch_size  = nnunet_cfg["patch_size"]
    norm_mean   = nnunet_cfg["norm_mean"]
    norm_std    = nnunet_cfg["norm_std"]
    num_classes = nnunet_cfg["num_classes"]

    print(f"  patch_size  : {patch_size}")
    print(f"  norm_mean   : {norm_mean:.4f}")
    print(f"  norm_std    : {norm_std:.4f}")
    print(f"  num_classes : {num_classes}")

    # ------------------------------------------------------------------
    # 2. Costruisci il modello e carica i pesi del fold
    # ------------------------------------------------------------------
    print("\n[2/4] Costruzione modello nnUNet...")
    device = torch.device(args.device)

    model = build_model_from_plans(
        plans         = nnunet_cfg["plans"],
        configuration = args.configuration,
        num_classes   = num_classes,
        device        = device,
    )

    # Checkpoint nnUNet v2: nnUNet_results/<Task>/fold_X/checkpoint_best.pth
    checkpoint_path = os.path.join(
        args.model_path, f"fold_{args.fold}", "checkpoint_best.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")

    print(f"  Carico pesi da: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # nnUNet v2 salva i pesi sotto la chiave 'network_weights'
    state_dict = checkpoint.get("network_weights", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print("  Modello pronto.")

    # ------------------------------------------------------------------
    # 3. Carica il dataset paziente
    # ------------------------------------------------------------------
    print("\n[3/4] Caricamento video paziente...")
    subject_path = os.path.join(args.dataset_path, args.subject)
    patient_dataset = PatientDataset(subject_path=subject_path)
    print(f"  Zone trovate: {patient_dataset.zones}")

    # Cartella output
    inference_folder = os.path.join(os.path.dirname(args.dataset_path), "inference_segmentation")
    subject_folder   = os.path.join(inference_folder, args.subject)
    os.makedirs(subject_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # 4. Inferenza frame-per-frame per ogni zona
    # ------------------------------------------------------------------
    print("\n[4/4] Inferenza...\n")
    BATCH_SIZE = 15  # frame per batch

    for i in range(len(patient_dataset)):
        video_frames, label, zone = patient_dataset[i]   # (F, H, W) uint8
        total_frames = video_frames.shape[0]
        print(f"  Zona {zone} | label={label} | frames={total_frames}")

        all_frames_dict = {
            "frame":           [],
            "frame_pleura":    [],
            "frame_ribs":      [],
            "centroid_pleura": [],
            "centroid_ribs":   [],
        }

        for start in range(0, total_frames, BATCH_SIZE):
            end   = min(start + BATCH_SIZE, total_frames)
            batch_np = video_frames[start:end]  # (B, H, W)

            # --- preprocessing: ogni frame indipendente (nnUNet 2D) ---
            preprocessed = []
            for frame in batch_np:
                t = preprocess_frame(frame, patch_size, norm_mean, norm_std)  # (1,1,H,W)
                preprocessed.append(t)

            # Concatena lungo dim batch → (B, 1, H, W)
            batch_tensor = torch.cat(preprocessed, dim=0).to(device)

            t0 = time.time()
            with torch.no_grad():
                pred = model(batch_tensor)   # (B, num_classes, H, W) — logits
            print(f"    Batch [{start}:{end}] — inference: {time.time()-t0:.3f}s")

            # --- visualizzazione: usa la funzione esistente ---
            # visualize_inference si aspetta (B,C,H,W) per il video e logits per pred
            # passiamo batch_tensor (float, normalizzato) e pred (logits)
            batch_dict = visualize_inference(batch_tensor, pred)

            for key in all_frames_dict:
                all_frames_dict[key].extend(batch_dict[key])

        # GIF per questa zona
        gif_name = f"{args.subject}_{zone}_label_{label}.gif"
        gif_path = os.path.join(subject_folder, gif_name)
        make_gif(all_frames_dict, output_path=gif_path, fps=patient_dataset.fps)
        print(f"  ✓ GIF salvata: {gif_path}")

    print("\nInferenza completata.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference nnUNet v2 (2D, grayscale) su video ecografici polmonari"
    )
    parser.add_argument("--model_path",        type=str, required=True,
                        help="Cartella nnUNet_results/<Task>/ che contiene fold_X/")
    parser.add_argument("--plans_path",        type=str, required=True,
                        help="Path al file plans.json (es. nnUNet_preprocessed/<Task>/plans.json)")
    parser.add_argument("--dataset_json_path", type=str, required=True,
                        help="Path al file dataset.json del task nnUNet")
    parser.add_argument("--dataset_path",      type=str, required=True,
                        help="Cartella root del dataset pazienti")
    parser.add_argument("--subject",           type=str, required=True,
                        help="Nome della cartella del paziente")
    parser.add_argument("--fold",              type=int, default=0,
                        help="Fold da usare (default: 0)")
    parser.add_argument("--configuration",     type=str, default="2d",
                        help="Configurazione nnUNet (default: '2d')")
    parser.add_argument("--device",            type=str, default="cuda",
                        help="Device torch: 'cuda' o 'cpu'")

    args = parser.parse_args()
    main(args)