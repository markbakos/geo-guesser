from keras import preprocessing, applications, models
import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from PIL import Image

def load_and_process_image(image_path: str, target_size=(224,224)):
    img = preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return applications.resnet50.preprocess_input(img_array)

def predict_location(image_path: str, model_path: str = "model/checkpoints/best_model.keras"):
    try:
        model = models.load_model(
            model_path,
            custom_objects={
                'haversine_loss': lambda y_true, y_pred: tf.math.reduce_mean(tf.math.abs(y_true - y_pred)),
                'location_accuracy': lambda y_true, y_pred: tf.math.reduce_mean(tf.cast(tf.abs(y_true - y_pred) <= 100, tf.float32))
            }
        )

        processed_image = load_and_process_image(image_path)

        predicted_coords = model.predict(processed_image)[0]
        latitude, longitude = predicted_coords

        print("\nPrediction Results:")
        print(f"Latitude: {latitude:.4f}")
        print(f"Longitude: {longitude:.4f}")

        return latitude, longitude

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Predict location from an image")
    parser.add_argument('image_path', type=str, help="Path to the image file")
    parser.add_argument('--model_path', type=str, default="model/checkpoints/best_model.keras", help="Path to the saved model")
    args = parser.parse_args()

    if not Path(args.image_path).exists():
        print(f"Error: Image not found at {args.image_path}")
        return

    lat, lon = predict_location(args.image_path, args.model_path)

if __name__ == '__main__':
    main()