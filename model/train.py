import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import math
from torch.nn.functional import dropout
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torchvision import transforms
from PIL import Image
import os
from torch.optim.lr_scheduler import OneCycleLR
import wandb
from torch.cuda.amp import autocast, GradScaler

from architecture import LocationCNN
from dataset import get_data_loaders


def setup_paths():
    """Setup correct paths based on project structure"""
    current_dir = Path(__file__).parent
    app_dir = current_dir.parent
    collection_dir = app_dir / "collection"
    dataset_dir = collection_dir / "dataset"

    logs_dir = current_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    model_save_dir = current_dir / "saved_models"
    model_save_dir.mkdir(exist_ok=True)

    return {
        'dataset': dataset_dir,
        'images': dataset_dir / "images",
        'metadata': dataset_dir / "metadata",
        'logs': logs_dir,
        'models': model_save_dir
    }


def predict_location(model, image_path, device):
    """Predict coordinates from a single image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        coordinates = model(image)
        lat, lon = coordinates[0].cpu().numpy()

    return lat, lon


def haversine_loss(pred, target):
    """Loss function using Haversine distance"""
    R = 6371

    lat1, lon1 = pred[:, 0], pred[:, 1]
    lat2, lon2 = target[:, 0], target[:, 1]

    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    return torch.mean(R * c)

def calculate_metrics(pred, target):
    with torch.no_grad():
        distance_error = haversine_loss(pred, target)

        distances = []
        for p, t in zip(pred, target):
            lat1, lon1 = p
            lat2, lon2 = t

            R = 6371
            lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            distance = R * c

            distances.append(distance)

        distances = np.array(distances)
        acc_100km = np.mean(distances <= 100)
        acc_500km = np.mean(distances <= 500)
        acc_1000km = np.mean(distances <= 1000)

        return {
            'distance_error': distance_error.item(),
            'acc_100km': acc_100km,
            'acc_500km': acc_500km,
            'acc_1000km': acc_1000km,
            'median_error': np.median(distances)
        }

def train_model(paths, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
    wandb.init(project="location-prediction", config={
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "resnet50_backbone"
    })

    logging.basicConfig(
        filename=paths['logs'] / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = LocationCNN(dropout_rate=0.3, weights=True).to(device)

    train_loader, val_loader, test_loader = get_data_loaders(
        str(paths['dataset']),
        batch_size=batch_size
    )

    if not train_loader or not val_loader or not test_loader:
        logging.error("Failed to create data loaders. Check if dataset files exist.")
        return None, None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
    )

    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()
        train_metrics = []

        if epoch < epochs // 3:
            model._freeze_early_layers()
        else:
            model.unfreeze_backbone()

        for images, coordinates in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            coordinates = coordinates.to(device)

            with autocast():
                outputs = model(images)
                loss = haversine_loss(outputs, coordinates)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            batch_metrics = calculate_metrics(outputs.detach().cpu(), coordinates.cpu())
            train_metrics.append(batch_metrics)

        model.eval()
        val_metrics = []

        with torch.no_grad():
            for images, coordinates in val_loader:
                images = images.to(device)
                coordinates = coordinates.to(device)

                outputs = model(images)
                batch_metrics = calculate_metrics(outputs.cpu(), coordinates.cpu())
                val_metrics.append(batch_metrics)

        train_metrics_avg = {k: np.mean([m[k] for m in train_metrics]) for k in train_metrics[0]}
        val_metrics_avg = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}

        wandb.log({
            'epoch': epoch,
            'train_loss': train_metrics_avg['distance_error'],
            'val_loss': val_metrics_avg['distance_error'],
            'train_acc_500km': train_metrics_avg['acc_500km'],
            'val_acc_500km': train_metrics_avg['acc_500km'],
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if val_metrics_avg['distance_error'] < best_val_loss:
            best_val_loss = val_metrics_avg['distance_error']
            epochs_without_improvement = 0

            model_path = paths['models'] / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics': val_metrics_avg,
            }, model_path)
            logging.info(f"Saved new best model with validation loss: {best_val_loss:.2f} km")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered from no improvement at {epoch + 1} epochs")
                break

    return model, model_path


def main():
    paths = setup_paths()
    print("\nStarting training...")
    model, model_path = train_model(paths)

    if model is not None and model_path is not None:
        print(f"\nTraining completed! Best model saved to: {model_path}")

        model.eval()
        test_metrics = []

        test_loader = get_data_loaders(str(paths['dataset']))[2]
        device = next(model.parameters()).device

        with torch.no_grad():
            for images, coordinates in test_loader:
                images = images.to(device)
                coordinates = coordinates.to(device)

                outputs = model(images)
                batch_metrics = calculate_metrics(outputs.cpu(), coordinates.cpu())
                test_metrics.append(batch_metrics)

            test_metrics_avg = {k: np.mean([m[k] for m in test_metrics]) for k in test_metrics[0]}

            print("\nTest Metrics:")
            for metric, value in test_metrics_avg.items():
                print(f"{metric}: {value:.4f}")
    else:
        print("Training did not complete successfuly")

if __name__ == "__main__":
    main()