import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging

from torch.nn.functional import dropout
from tqdm import tqdm
import numpy as np
from datetime import datetime
from torchvision import transforms
from PIL import Image
import os

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


def train_model(paths, epochs=100, batch_size=32, learning_rate=0.001, patience=10):
    logging.basicConfig(
        filename=paths['logs'] / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = LocationCNN(dropout_rate=0.3).to(device)

    train_loader, val_loader, test_loader = get_data_loaders(
        str(paths['dataset']),
        batch_size=batch_size
    )

    if not train_loader or not val_loader or not test_loader:
        logging.error("Failed to create data loaders. Check if dataset files exist.")
        return None, None

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()
        train_losses = []

        for images, coordinates in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            coordinates = coordinates.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = haversine_loss(outputs, coordinates)

            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for images, coordinates in val_loader:
                images = images.to(device)
                coordinates = coordinates.to(device)

                outputs = model(images)
                loss = haversine_loss(outputs, coordinates)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        logging.info(f"Epoch {epoch + 1}/{epochs}")
        logging.info(f"Training Loss: {train_loss:.2f} km")
        logging.info(f"Validation Loss: {val_loss:.2f} km")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model_path = paths['models'] / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, model_path)
            logging.info(f"Saved new best model with validation loss: {val_loss:.2f} km")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logging.info(f"Early stopping triggered from no improvement at {epoch + 1} epochs")
                break

    return model, model_path


def main():
    paths = setup_paths()

    print("Checking paths:")
    for name, path in paths.items():
        exists = path.exists()
        print(f"{name}: {path} ({'exists' if exists else 'missing'})")

    print("\nStarting training...")
    model, model_path = train_model(paths)

    if model is not None and model_path is not None:
        print(f"\nTraining completed! Best model saved to: {model_path}")

        test_images = list(paths['images'].glob("*.jpg"))
        if test_images:
            print("\nTesting model prediction with a sample image...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            lat, lon = predict_location(model, test_images[0], device)
            print(f"Predicted location: {lat:.4f}°N, {lon:.4f}°E")
    else:
        print("\nTraining failed. Check the logs for details.")


if __name__ == "__main__":
    main()