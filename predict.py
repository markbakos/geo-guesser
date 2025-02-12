import json
from pathlib import Path
import numpy as np
from typing import Optional, Union, Dict, Tuple
from keras import preprocessing, applications, models
from model.metrics import haversine_loss, location_accuracy
import pickle
import cv2

class Predictor:
    def __init__(self, model_path: Optional[Path] = None):
        if model_path is None:
            model_path = Path("model/checkpoints/best_model/best_location_model.keras")

        if not model_path.exists():
            model_path = Path("model/checkpoints/best_model/best_overall_model.keras")

        if not model_path.exists():
            raise FileNotFoundError("No model found.")

        self.model = models.load_model(
            model_path,
            custom_objects={
                'haversine_loss': haversine_loss,
                'location_accuracy': location_accuracy
            }
        )

        encoder_path = Path("model/checkpoints/city_encoder.pkl")
        with open(encoder_path, 'rb') as f:
            self.city_encoder = pickle.load(f)

        stats_path = Path("model/checkpoints/coordinate_stats.pkl")
        with open(stats_path, 'rb') as f:
            self.coord_stats = pickle.load(f)

        from model.visualize import AttentionVisualizer
        self.visualizer = AttentionVisualizer(self.model)

    def preprocess_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        img = preprocessing.image.load_img(
            image_path,
            target_size=(224, 224),
        )
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return applications.efficientnet.preprocess_input(img_array)

    def denormalize_coordinates(self, normalized_coords: np.ndarray) -> Tuple[float, float]:
        lat = (normalized_coords[0] *
               (self.coord_stats['lat_max'] - self.coord_stats['lat_min']) +
               self.coord_stats['lat_min'])
        lon = (normalized_coords[1] *
               (self.coord_stats['lon_max'] - self.coord_stats['lon_min']) +
               self.coord_stats['lon_min'])
        return lat, lon

    def save_attention_heatmap(self, image_path: Union[str, Path], output_path: Optional[Path] = None) -> Path:
        if output_path is None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_attention.png"

        overlay = self.visualizer.visualize(str(image_path))

        cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        return output_path

    def predict(self, image_path: Union[str, Path], generate_heatmap: bool = False) -> Dict:
        image_path = Path(image_path)
        img_array = self.preprocess_image(image_path)

        city_probs, normalized_coordinates = self.model.predict(
            img_array,
            verbose=0
        )

        predicted_city_idx = np.argmax(city_probs[0])
        predicted_city = self.city_encoder.inverse_transform([predicted_city_idx])[0]

        lat, lon = self.denormalize_coordinates(normalized_coordinates[0])

        result = {
            'coordinates': {
                'latitude': float(lat),
                'longitude': float(lon)
            },
            'city': predicted_city,
            'city_confidence': float(city_probs[0][predicted_city_idx])
        }

        if generate_heatmap:
            heatmap_path = self.save_attention_heatmap(image_path)
            result['attention_heatmap_path'] = str(heatmap_path)

        return result


def main():
    import argparse

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
    parser.add_argument(
        '--generate_heatmap',
        action='store_true',
        help='Generate attention heatmap'
    )
    parser.add_argument(
        '--output_heatmap',
        type=str,
        help='Path to save the attention heatmap (optional)',
        default=None
    )

    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else None
    predictor = Predictor(model_path)

    try:
        result = predictor.predict(args.image_path, generate_heatmap=args.generate_heatmap)

        if args.generate_heatmap and args.output_heatmap:
            predictor.save_attention_heatmap(args.image_path, Path(args.output_heatmap))
            result['attention_heatmap_path'] = args.output_heatmap

        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()