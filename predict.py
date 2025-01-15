import json

from keras import preprocessing, applications, models
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from typing import Optional, Union, Dict

from model.metrics import haversine_loss, location_accuracy

class Predictor:
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path("model/checkpoints/best_model/best_location_model.keras")

        if not model_path.exists():
            model_path = Path("model/checkpoints/best_model/best_overall_model.keras")

        if not model_path.exists():
            raise FileNotFoundError(
                "No model found."
            )

        self.model = models.load_model(
            model_path,
            custom_objects={
                'haversine_loss': haversine_loss,
                'location_accuracy': location_accuracy
            }
        )

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = preprocessing.image.load_img(
            image_path,
            target_size=(224,224),
        )
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return applications.efficientnet.preprocess_input(img_array)

    def predict(self, image_path: Union[str, Path]) -> Dict:
        image_path = Path(image_path)
        img_array = self.preprocess_image(image_path)

        region_probs, scene_probs, coordinates = self.model.predict(
            img_array,
            verbose=0
        )

        predicted_region = np.argmax(region_probs[0])
        predicted_scene = np.argmax(scene_probs[0])

        return {
            'coordinates': (float(coordinates[0][0]), float(coordinates[0][1])),
            'region': int(predicted_region),
            'region_confidence': float(region_probs[0][predicted_region]),
            'scene': int(predicted_scene),
            'scene_confidence': float(scene_probs[0][predicted_scene])
        }

def main():
    parser = argparse.ArgumentParser(description='Predict location from image')
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to the image file'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to the model file (optional)',
        default=None
    )

    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else None
    predictor = Predictor(model_path)

    try:
        result = predictor.predict(args.image_path)
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        exit(1)

if __name__ == '__main__':
    main()