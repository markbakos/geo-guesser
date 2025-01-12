import logging
import torch.cuda
from pathlib import Path
from model.architecture import LocationCNN
from torchvision import transforms
from PIL import Image


class LocationPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LocationCNN().to(self.device)

        model_path = Path(__file__).parent / "model" / "saved_models" / "best_model.pth"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                coordinates = self.model(image_tensor)
                lat, lon = coordinates[0].cpu().numpy()

            return {
                'latitude': float(lat),
                'longitude': float(lon),
                'success': True
            }
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        import sys
        if len(sys.argv) != 2:
            print("Usage: python predict.py <image_path>")
            sys.exit(1)

        image_path = sys.argv[1]

        predictor = LocationPredictor()
        result = predictor.predict(image_path)

        if result['success']:
            print(f"\nPredicted location:")
            print(f"Latitude: {result['latitude']}°N")
            print(f"Longitude: {result['longitude']}°E")
        else:
            print(f"Error: {result['error']}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()