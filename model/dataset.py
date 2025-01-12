import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import logging


class LocationDataset(Dataset):
    def __init__(self, metadata_path, images_dir, transform=None):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = self.images_dir / f"processed_{row['filename']}"

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        coordinates = torch.tensor([
            row['latitude'],
            row['longitude']
        ], dtype=torch.float32)

        return image, coordinates


def get_data_loaders(base_path, batch_size=32, num_workers=4):
    base_path = Path(base_path)
    images_dir = base_path / "images"
    metadata_dir = base_path / "metadata"

    if not (metadata_dir / "train_metadata.csv").exists():
        logging.info("Split metadata files not found. Please run prepare_data.py first")
        return None, None, None

    try:
        train_dataset = LocationDataset(
            metadata_dir / "train_metadata.csv",
            images_dir
        )

        val_dataset = LocationDataset(
            metadata_dir / "val_metadata.csv",
            images_dir
        )

        test_dataset = LocationDataset(
            metadata_dir / "test_metadata.csv",
            images_dir
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error creating data loaders: {str(e)}")
        return None, None, None