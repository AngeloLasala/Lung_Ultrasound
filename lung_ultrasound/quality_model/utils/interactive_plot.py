"""
export_data.py

Lancia questo script nella stessa directory del tuo progetto:
    python export_data.py

Output: ultrasound_data.json  (nella stessa cartella di scatter.html)
"""

import os
import io
import base64
import json
import tqdm
import torch
import numpy as np
from PIL import Image

from lung_ultrasound.quality_model.cfg import cfg
from lung_ultrasound.quality_model.dataset.dataset import JointTransform2D, LungDataset
from lung_ultrasound.quality_model.utils.prototype import get_parameters_from_mask, is_gold_standard

# ── GS bounds ────────────────────────────────────────────────────────────────
bounds = {
    'p1': [-0.15,  0.15], 'p2': [-0.45,  0.20],
    'p3': [ 0.90,  1.00], 'p4': [-0.15,  0.15],
    'r1': [-0.15,  0.15], 'r2': [-0.20,  0.20],
    'r3': [ 0.90,  1.00],
}

# ── utility ──────────────────────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None

# ── dataset ──────────────────────────────────────────────────────────────────
transform = JointTransform2D(
    img_size=cfg.img_size, low_img_size=cfg.img_size,
    ori_size=cfg.img_size, crop=cfg.crop,
    p_flip=0, p_rota=0, p_scale=0, p_gaussn=0,
    p_contr=0, p_gama=0, p_distor=0,
    color_jitter_params=cfg.color_jitter_params,
    long_mask=cfg.long_mask,
)

dataset_list = []
for split in ['train', 'val']:
    ds = LungDataset(
        dataset_path=os.path.join(cfg.main_path, cfg.dataset),
        img_size=cfg.img_size,
        fold_cv=cfg.fold_cv,
        splitting_json=cfg.splitting,
        split=split,
        joint_transform=transform,
        one_hot_mask=False,
    )
    dataset_list.append(ds)

dataset = torch.utils.data.ConcatDataset(dataset_list)

# ── esporta ──────────────────────────────────────────────────────────────────
samples = []
gs_count = 0

for data in tqdm.tqdm(dataset):
    mask   = data['label'].detach().cpu().numpy()
    params = get_parameters_from_mask(mask)
    gs     = is_gold_standard(params, bounds)
    gs_count += gs

    # immagine → base64 PNG
    img_np = data['image'].permute(1, 2, 0).numpy()
    img_np = img_np[..., 0] if img_np.shape[2] == 1 else img_np
    img_np = ((img_np - img_np.min()) /
              (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
    mode = 'L' if img_np.ndim == 2 else 'RGB'
    pil  = Image.fromarray(img_np, mode=mode).resize((256, 256))
    buf  = io.BytesIO()
    pil.save(buf, format='PNG')
    b64  = base64.b64encode(buf.getvalue()).decode()

    samples.append({
        'subject': str(data['subject']),
        'zone':    str(data['zone']),
        'is_gs':   gs,
        'pleura': {
            'p1': safe_float(params['pleura']['p1']),
            'p2': safe_float(params['pleura']['p2']),
            'p3': safe_float(params['pleura']['p3']),
            'p4': safe_float(params['pleura']['p4']),
            'centroid_x': safe_float(params['pleura']['centroid_x']),
            'centroid_y': safe_float(params['pleura']['centroid_y']),
        },
        'ribs': {
            'r1': safe_float(params['ribs']['r1']),
            'r2': safe_float(params['ribs']['r2']),
            'r3': safe_float(params['ribs']['r3']),
            'centroid_x': safe_float(params['ribs']['centroid_x']),
            'centroid_y': safe_float(params['ribs']['centroid_y']),
        },
        'img_b64': f'data:image/png;base64,{b64}',
    })

html_path   = r"C:\Users\lasal\Documents\Lung_Ultrasound\lung_ultrasound\quality_model\utils\html_files"
output_path = os.path.join(html_path, 'ultrasound_data.json')
with open(output_path, 'w') as f:
    json.dump(samples, f)

print(f"Salvati {len(samples)} campioni in '{output_path}'")
print(f"Gold standard: {gs_count} / {len(samples)} ({100*gs_count/len(samples):.1f}%)")