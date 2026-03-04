"""
Create dataset from annotations in Label-Studio for the segmentation task od Rib shadow and Pleura
"""
import os 
import argparse
import cv2 
import tqdm
import json
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  Label Studio RLE decoder (thi-ng/rle-pack)
#  Source: https://github.com/thi-ng/umbrella/blob/develop/packages/rle-pack/src/index.ts
# ─────────────────────────────────────────────

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i: self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """From bytes array to bit by position num."""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """Get bit string from bytes data."""
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: list, width: int, height: int) -> np.ndarray:
    """
    Converte un RLE di Label Studio (formato brush, thi-ng/rle-pack) in una maschera 2D.

    Il flat array decodificato rappresenta un'immagine RGBA (4 canali).
    La maschera effettiva si trova nel canale alpha (indice 3).

    Args:
        rle:    lista di interi (campo 'rle' dal JSON di Label Studio)
        width:  original_width dal risultato dell'annotazione
        height: original_height dal risultato dell'annotazione

    Returns:
        np.ndarray di shape (height, width), dtype uint8, valori 0-255
    """
    # 1. Lista di int → bytes → stringa di bit
    rle_bytes = bytes(rle)
    bit_string = bytes2bit(rle_bytes)

    # 2. Leggi header: num valori, word size, rle_sizes
    stream = InputStream(bit_string)
    num       = stream.read(32)
    word_size = stream.read(5) + 1
    rle_sizes = [stream.read(4) + 1 for _ in range(4)]

    # 3. Decodifica corpo
    flat = np.zeros(num, dtype=np.uint8)
    i = 0
    while i < num:
        x = stream.read(1)
        j = i + 1 + stream.read(rle_sizes[stream.read(2)])
        if x:
            val = stream.read(word_size)
            flat[i:j] = val
            i = j
        else:
            while i < j:
                flat[i] = stream.read(word_size)
                i += 1

    # 4. Reshape RGBA → prendi canale alpha (la maschera)
    mask = np.reshape(flat, [height, width, 4])[:, :, 3]
    return mask  # shape (height, width), valori 0-255

def fill_mask_regions(mask: np.ndarray, close_kernel: int = 1) -> np.ndarray:
    """
    Prende una maschera binaria con contorni disegnati e riempie le regioni chiuse.

    Args:
        mask:         np.ndarray (height, width), dtype uint8, valori 0 o 1
        close_kernel: dimensione kernel morfologico per chiudere eventuali gap

    Returns:
        np.ndarray (height, width), dtype uint8, valori 0 o 1
    """
    # 1. Chiudi piccoli gap nei contorni
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. Flood fill dall'angolo (0,0) → raggiunge tutto l'esterno
    flood = closed.copy()
    seed_mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, seed_mask, seedPoint=(0, 0), newVal=1)

    # 3. Ciò che il flood non ha raggiunto = interno delle regioni
    interior = (flood == 0).astype(np.uint8)

    # 4. Unisci contorno originale + interno
    result = np.clip(mask + interior, 0, 1).astype(np.uint8)
    return result

def get_centroids(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Restituisce le coordinate (row, col) del centroide di ogni regione connessa.

    Args:
        mask: np.ndarray (height, width), dtype uint8, valori 0 o 1

    Returns:
        lista di tuple (row, col) per ogni regione connessa
    """
    num_labels, labels = cv2.connectedComponents(mask)
    
    centroids = []
    for label in range(1, num_labels):  # 0 è il background
        region = (labels == label)
        rows, cols = np.where(region)
        centroid_r = int(rows.mean())
        centroid_c = int(cols.mean())
        centroids.append((centroid_r, centroid_c))
    
    return centroids

def main(args):
    """
    Read export file and return the dataset folder with images (already have from frames_extrapolation) and labels
    """
    ## Read raw
    subjects_path = os.path.join(args.main_path, args.frames_path)
    subjects_list = os.listdir(subjects_path)

    ## Check valid Subject
    print('Checking valid subject...')
    subjects_valid = []
    for subject in subjects_list:
        subject_path = os.path.join(subjects_path, subject)

        # controlla che sia una directory
        if not os.path.isdir(subject_path):
            continue

        images_path = os.path.join(subject_path, "images")
        # cerca un file json nella cartella del subject
        json_files = [f for f in os.listdir(subject_path) if f.endswith(".json")]

        if not os.path.exists(images_path):
            print(f"[WARNING] {subject} has no 'images' folder")
            continue

        if len(json_files) == 0:
            print(f"[WARNING] {subject} has no JSON annotation file")
            continue

        if len(json_files) > 1:
            print(f"[WARNING] {subject} has multiple JSON files, taking the first one")

        annotation_path = os.path.join(subject_path, json_files[0])

        subjects_valid.append({
            "name": subject,
            "images_path": images_path,
            "annotation_path": annotation_path
        })

    print(f"Found {len(subjects_valid)} valid subjects")
    print()

    print('Extrapolate annotations for valid subjects...')
    n = 0
    pleuras_list = []
    ribs_shadow_list = []
    for subject in subjects_valid:
        name = subject['name']
        images_path = subject['images_path']
        annotation_path = subject['annotation_path']
        print(name)
       
        ## create labels path to save annotations as masks
        labels_path = os.path.join(subjects_path, name,  "labels")
        os.makedirs(labels_path, exist_ok=True)

        # read annotation path with json
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        ## check if the number of annotations is equals to the number of images
        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
        images_list = [f for f in os.listdir(images_path) if f.lower().endswith(valid_ext)]
        n_images = len(images_list)
        n_annotations = len(annotations)

        if n_images != n_annotations:
            raise ValueError(
                f"Mismatch in subject '{name}': "
                f"{n_images} images found but {n_annotations} annotations found."
            )

        # read annotation
        for ann in annotations:
            n += 1
            image_name = ann['image'].split('-')[-1]
            
            # read image with pillow
            image = Image.open(os.path.join(images_path, image_name))
            image = np.array(image)
            h, w = image.shape[0], image.shape[1]

            # initial mask (vuote di default)
            pleura_mask = np.zeros((h, w), dtype=np.uint8)
            ribs_shadow_mask = np.zeros((h, w), dtype=np.uint8)

            # check esistenza 'tag'
            if 'tag' in ann and ann['tag']:
                segmentation_rle = ann['tag']

                ## loop over each segmentation mask in annotation
                for mask_ann in segmentation_rle:
                    class_name = mask_ann['brushlabels'][0]
                    original_w = mask_ann['original_width']
                    original_h = mask_ann['original_height']
                    rle = mask_ann['rle']
                    
                    if class_name == 'Pleura':
                        pleura_mask = rle_to_mask(rle, original_w, original_h)
                        pleura_mask = fill_mask_regions(pleura_mask)
                        pleura_centroids = get_centroids(pleura_mask)
                        for c in pleura_centroids:
                            pleuras_list.append(c)
                        
                    if class_name == 'Coste':
                        ribs_shadow_mask = rle_to_mask(rle, original_w, original_h)
                        ribs_shadow_mask = fill_mask_regions(ribs_shadow_mask)
                        ribs_shadow_mask_centroids = get_centroids(ribs_shadow_mask)
                        for c in ribs_shadow_mask_centroids:
                            ribs_shadow_list.append(c)

            # stack (anche se sono rimaste vuote)
            labels_mask = np.stack((pleura_mask, ribs_shadow_mask), axis=0)

            # converting in a single vecto 0=background, 1=pleura , 2=ribs' shadow
            pleura_mask = labels_mask[0]
            ribs_shadow_mask = labels_mask[1]

            single_channel_mask = np.zeros((pleura_mask.shape[0], pleura_mask.shape[1]), dtype=np.uint8)
            single_channel_mask[pleura_mask > 0] = 1
            single_channel_mask[ribs_shadow_mask > 0] = 2

            if len(image_name.split('.')) == 2:
                saving_path = os.path.join(labels_path, f"{image_name.split('.')[0]}.npy")
            else:
                name = image_name.split('.')[0] + '.' + image_name.split('.')[1]
                saving_path = os.path.join(labels_path, f"{name}.npy")

            np.save(saving_path, single_channel_mask)

            ##################################################
            # plt.figure(figsize=(12, 4))

            # plt.subplot(1, 3, 1)
            # plt.imshow(image, cmap='gray')
            # plt.title("Original Image")
            # plt.imshow(single_channel_mask, cmap='hot', alpha=0.3)
            # plt.axis('off')

            # plt.subplot(1, 3, 2)
            # plt.imshow(pleura_mask, cmap='Reds')
            # pleura_centroids = get_centroids(pleura_mask)
            # for c in pleura_centroids:
            #     plt.scatter(c[1], c[0], c='orange')
            # plt.axis('off')

            # plt.subplot(1, 3, 3)
            # plt.imshow(ribs_shadow_mask, cmap='Blues')
            # ribs_shadow_mask_centroids = get_centroids(ribs_shadow_mask)
            # for c in ribs_shadow_mask_centroids:
            #     plt.scatter(c[1], c[0], c='yellow')
            # plt.title("Ribs Shadow Mask")
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
            #################################################

        print(f'Annotated images: {n}')
        print(f'Pleuras: {len(pleuras_list)}')
        print(f'Rib shadows: {len(ribs_shadow_list)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extrapolate frames from video')
    parser.add_argument('--main_path', help='Path to the main folder, i.e. ../OpenPOCUS')
    parser.add_argument('--frames_path', help='Path to the output folder where frames will be saved, i.e. DATA_extrapolate_frames/')
    args = parser.parse_args()

    main(args)