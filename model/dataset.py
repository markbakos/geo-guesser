import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import logging
import albumentations as A
import numpy as np

class LocationDataset(Dataset):
    def __init__(self, metadata_path, images_dir, transform=None, train_mode=False):
        self.metadata_path = Path(metadata_path)
        self.images_dir = Path(images_dir)
        self.train_mode = train_mode

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)

        if transform is None:
            if train_mode:
                self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.2),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                    transforms.RandomAffine(
                        degrees=10,
                        translate=(0.1, 0.1),
                        scale=(0.9, 1.1)
                    ),
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=3)
                    ], p=0.3),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.2),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                self.albu_transform = A.Compose([
                    A.OneOf([
                        A.RandomBrightnessContrast(p=1),
                        A.RandomGamma(p=1),
                        A.CLAHE(p=1),
                    ], p=0.5),
                    A.OneOf([
                        A.GaussNoise(p=1),
                        A.ISONoise(p=1),
                        A.MultiplicativeNoise(p=1),
                    ], p=0.3),
                    A.OneOf([
                        A.Sharpen(p=1),
                        A.Blur(p=1),
                        A.MotionBlur(p=1),
                    ], p=0.3),
                    A.OneOf([
                        A.RandomShadow(p=1),
                        A.RandomSunFlare(p=1),
                        A.RandomRain(p=1),
                    ], p=0.2),
                ])

            else:
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

        if self.train_mode and self.albu_transform:
            image_np = np.array(image)
            augmented = self.albu_transform(image=image_np)
            image = Image.fromarray(augmented['image'])

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
            images_dir,
            train_mode=True
        )

        val_dataset = LocationDataset(
            metadata_dir / "val_metadata.csv",
            images_dir,
            train_mode=False
        )

        test_dataset = LocationDataset(
            metadata_dir / "test_metadata.csv",
            images_dir,
            train_mode=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error creating data loaders: {str(e)}")
        return None, None, None