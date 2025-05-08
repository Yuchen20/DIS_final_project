import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tifffile import imread

class OriDataset2(Dataset):
    def __init__(self, configs):
        self.cell_type = configs['cell_type']
        self.purpose = configs['purpose']
        self.pixel_cutoff = 0.5
        self.data_root = "/home/ym429/rds/hpc-work/dissertation"  # Absolute path
        self.image_size = configs['image_size']
        
        # Define plate identifiers based on cell type and purpose
        if self.cell_type == 'u2os':
            if self.purpose == 'train':
                self.plates = ["BR00116995", "BR00117024"]  # U2OS Train
            elif self.purpose == 'val':
                self.plates = ["BR00117025"]  # U2OS Validate
            elif self.purpose == 'test':
                self.plates = ["BR00117026", "BR00118045"]  # U2OS Test (including crispr)
        elif self.cell_type == 'a549':
            if self.purpose == 'train':
                self.plates = ["BR00116991", "BR00116992"]  # A549 Train
            elif self.purpose == 'val':
                self.plates = ["BR00116993"]  # A549 Validate
            elif self.purpose == 'test':
                self.plates = ["BR00116994", "BR00118041"]  # A549 Test (including crispr)
        elif self.cell_type == 'both':
            if self.purpose == 'train':
                self.plates = ["BR00116991", "BR00116992", "BR00116995", "BR00117024"]  # All Train
            elif self.purpose == 'val':
                self.plates = ["BR00116993", "BR00117025"]  # All Validate
            elif self.purpose == 'test':
                self.plates = ["BR00116994", "BR00117026", "BR00118041", "BR00118045"]  # All Test (including crispr)

        # Load image paths from CSV files
        self.imlist = []
        for plate in self.plates:
            csv_path = os.path.join(self.data_root, f"unique_paths_{plate}.csv")
            with open(csv_path, 'r', newline='') as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        self.imlist.append((os.path.join(self.data_root, plate, 'Images'), row[0].strip()))

        if self.purpose == 'val':
            self.imlist = self.imlist[:10]

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])

    def normalize(self, img):
        lower_threshold = np.percentile(img, self.pixel_cutoff)
        upper_threshold = np.percentile(img, 100 - self.pixel_cutoff)
        img_clipped = np.clip(img, lower_threshold, upper_threshold)
        img_clipped = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped)) * 255
        return np.array(img_clipped, dtype=np.uint8)

    def __getitem__(self, idx):
        base_dir, prefix = self.imlist[idx]
        bf = []
        
        # Load label channels (source channels: [6,7,8])
        for i in [6,7,8]:
            img = imread(os.path.join(base_dir, f"{prefix}-ch%isk1fk1fl1.tiff" % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            bf.append(img_preprocessed)

        label_tfm = torch.cat(bf)

        # Load output channels (target channels: [1,2,5])
        output = []
        for i in [1,2,5]:
            img = imread(os.path.join(base_dir, f"{prefix}-ch%isk1fk1fl1.tiff" % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            output.append(img_preprocessed)

        img_tfm = torch.cat(output)

        if self.purpose == 'test':
            return_d = {'gt': img_tfm, 'lq': label_tfm, 'filename': os.path.join(base_dir, prefix)}
        else:
            return_d = {'gt': img_tfm, 'lq': label_tfm}
        return return_d

    def __len__(self):
        return len(self.imlist)

class OriDataset2_3channel(Dataset):
    def __init__(self, configs):
        self.cell_type = configs['cell_type']
        self.purpose = configs['purpose']
        self.pixel_cutoff = 0.5
        self.data_root = "/home/ym429/rds/hpc-work/dissertation"  # Absolute path
        self.image_size = configs['image_size']
        
        # Define plate identifiers based on cell type and purpose
        if self.cell_type == 'u2os':
            if self.purpose == 'train':
                self.plates = ["BR00116995", "BR00117024"]  # U2OS Train
            elif self.purpose == 'val':
                self.plates = ["BR00117025"]  # U2OS Validate
            elif self.purpose == 'test':
                self.plates = ["BR00117026", "BR00118045"]  # U2OS Test (including crispr)
        elif self.cell_type == 'a549':
            if self.purpose == 'train':
                self.plates = ["BR00116991", "BR00116992"]  # A549 Train
            elif self.purpose == 'val':
                self.plates = ["BR00116993"]  # A549 Validate
            elif self.purpose == 'test':
                self.plates = ["BR00116994", "BR00118041"]  # A549 Test (including crispr)
        elif self.cell_type == 'both':
            if self.purpose == 'train':
                self.plates = ["BR00116991", "BR00116992", "BR00116995", "BR00117024"]  # All Train
            elif self.purpose == 'val':
                self.plates = ["BR00116993", "BR00117025"]  # All Validate
            elif self.purpose == 'test':
                self.plates = ["BR00116994", "BR00117026", "BR00118041", "BR00118045"]  # All Test (including crispr)

        # Load image paths from CSV files
        self.imlist = []
        for plate in self.plates:
            csv_path = os.path.join(self.data_root, f"unique_paths_{plate}.csv")
            with open(csv_path, 'r', newline='') as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        self.imlist.append((os.path.join(self.data_root, plate, 'Images'), row[0].strip()))

        if self.purpose == 'val':
            self.imlist = self.imlist[:10]

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5], [0.5]),
        ])

    def normalize(self, img):
        lower_threshold = np.percentile(img, self.pixel_cutoff)
        upper_threshold = np.percentile(img, 100 - self.pixel_cutoff)
        img_clipped = np.clip(img, lower_threshold, upper_threshold)
        img_clipped = (img_clipped - np.min(img_clipped)) / (np.max(img_clipped) - np.min(img_clipped)) * 255
        return np.array(img_clipped, dtype=np.uint8)

    def __getitem__(self, idx):
        base_dir, prefix = self.imlist[idx]
        bf = []
        
        # Load label channels (source channels: [6,7,8])
        for i in [6,7,8]:
            img = imread(os.path.join(base_dir, f"{prefix}-ch%isk1fk1fl1.tiff" % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            bf.append(img_preprocessed)

        label_tfm = torch.cat(bf)

        # Load output channels (target channels: [1,2,5])
        output = []
        for i in [1,2,5]:
            img = imread(os.path.join(base_dir, f"{prefix}-ch%isk1fk1fl1.tiff" % (i)))
            img = self.normalize(img)
            img_preprocessed = self.transform(Image.fromarray(img))
            output.append(img_preprocessed)

        img_tfm = torch.cat(output)

        if self.purpose == 'test':
            return_d = {'gt': img_tfm, 'lq': label_tfm, 'filename': os.path.join(base_dir, prefix)}
        else:
            return_d = {'gt': img_tfm, 'lq': label_tfm}
        return return_d

    def __len__(self):
        return len(self.imlist) 