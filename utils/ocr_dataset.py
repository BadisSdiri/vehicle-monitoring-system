import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import string

class LicensePlateDataset(Dataset):
    def __init__(self, images_folder, detection_csv, recognition_csv, transform=None):
        self.images_folder = images_folder
        self.detection_data = pd.read_csv(detection_csv)
        self.recognition_data = pd.read_csv(recognition_csv)
        self.transform = transform

        # Merge detection and recognition data
        self.data = pd.merge(
            self.detection_data, 
            self.recognition_data, 
            on="img_id"
        )

        # Define character-to-index and index-to-character mappings
        self.chars = string.ascii_uppercase + string.digits + " "  # Include letters, digits, and space
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.images_folder, row["img_id"])
        img = Image.open(img_path).convert("RGB")

        # Crop using bounding box
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        # Apply transformations if specified
        if self.transform:
            cropped_img = self.transform(cropped_img)

        # Text label (convert to indices)
        text = row["text"]
        label = [self.char_to_idx[char] for char in text]

        return cropped_img, torch.tensor(label)
