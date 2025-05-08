import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from noise_scheduling import DiffusionScheduler, CFG

def process_and_resize_image(filepath, img_size=(512, 512)):
    """
    Load a TIFF image, apply percentile clipping for contrast, resize, and return as numpy array.
    Handles edge cases where image might be uniform or have invalid values.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {filepath}")
    
    # Handle NaN and inf values
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get percentiles
    p5, p95 = np.percentile(img, [5, 95])
    
    # Handle case where p5 equals p95 (uniform image)
    if p5 == p95:
        # If image is uniform, return zeros
        return np.zeros(img_size, dtype=np.uint8)
    
    # Clip and normalize
    img = np.clip(img, p5, p95)
    img = ((img - p5) / (p95 - p5) * 255).astype(np.uint8)
    
    # Resize
    return cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

class DiffusionDataset(Dataset):
    """
    Dataset that reads partial paths from CSVs, loads multi-channel TIFFs, and returns source/target tensors.
    """
    def __init__(self, csv_file_list, images_dir = '/home/ym429/rds/hpc-work/dissertation', source_channels=[1,2,3], target_channels=[4,5,6], img_size=(512,512), transform=None, normalize=True):
        self.samples = []  # list of (directory, partial_path)
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.img_size = img_size
        self.transform = transform
        self.normalize = normalize

        for csv_path in csv_file_list:
            with open(csv_path, 'r', newline='') as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        self.samples.append((os.path.join(images_dir, csv_path.split('/')[-1].split('_')[-1][:-4], 'Images'), row[0].strip()))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        base_dir, prefix = self.samples[idx]
        def load_chs(ch_list):
            imgs = []
            for ch in ch_list:
                path = os.path.join(base_dir, f"{prefix}-ch{ch}sk1fk1fl1.tiff")
                imgs.append(process_and_resize_image(path, self.img_size))
            arr = np.stack(imgs, axis=0)
            tensor = torch.from_numpy(arr).float()
            if self.normalize:
                tensor = tensor / 255.0
            if self.transform:
                tensor = self.transform(tensor)
            return tensor
        source = load_chs(self.source_channels)
        target = load_chs(self.target_channels)
        return source, target
    def load_all_channels(self, idx, channels=None):
        base_dir, prefix = self.samples[idx]
        chs = channels if channels is not None else list(range(1,9))
        imgs = []
        for ch in chs:
            path = os.path.join(base_dir, f"{prefix}-ch{ch}sk1fk1fl1.tiff")
            imgs.append(process_and_resize_image(path, self.img_size))
        arr = np.stack(imgs, axis=2)
        tensor = torch.from_numpy(arr).float()
        if self.normalize:
            tensor = tensor / 255.0
        return tensor

def get_diffusion_dataloader(csv_file_list, batch_size=4, source_channels=[1,2,3], target_channels=[4,5,6], img_size=(512,512), shuffle=True, num_workers=4):
    dataset = DiffusionDataset(csv_file_list, source_channels, target_channels, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def visualize_image_channels(img, title=None, cmap="viridis"):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] <= 8:
        img = np.transpose(img, (1,2,0))
    c = img.shape[2]
    fig, axes = plt.subplots(1, c, figsize=(3*c,3))
    for i in range(c):
        axes[i].imshow(img[:,:,i], cmap=cmap)
        axes[i].axis('off')
        axes[i].set_title(f"Channel {i+1}")
    if title: plt.suptitle(title)
    plt.tight_layout()
    plt.show()


