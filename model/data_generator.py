import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras import Sequential, utils, layers, preprocessing, applications
import pickle

class DataGenerator(utils.Sequence):
    """Creates batches of preprocessed images for training the model"""
    def __init__(self, metadata_path: Path, images_dir: Path, batch_size: int = 32, augment: bool = False, shuffle: bool = True):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))

        self.lat_min, self.lat_max = self.metadata['latitude'].min(), self.metadata['latitude'].max()
        self.lon_min, self.lon_max = self.metadata['longitude'].min(), self.metadata['longitude'].max()

        self.city_encoder = LabelEncoder()
        self.metadata['city_encoded'] = self.city_encoder.fit_transform(self.metadata['city'])

        stats = {
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
            'lon_min': self.lon_min,
            'lon_max': self.lon_max
        }

        encoder_path = Path("model/checkpoints/city_encoder.pkl")
        stats_path = Path("model/checkpoints/coordinate_stats.pkl")
        encoder_path.parent.mkdir(exist_ok=True, parents=True)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.city_encoder, f)
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        if self.augment:
            self.aug_layer = Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2),
            ])

        self.on_epoch_end()

    def normalize_coordinates(self, lat, lon):
        """Normalizes lat and lon coordinate values to a [0-1] range."""
        lat_norm = (lat - self.lat_min) / (self.lat_max - self.lat_min)
        lon_norm = (lon - self.lon_min) / (self.lon_max - self.lon_min)
        return np.array([lat_norm, lon_norm])

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        """Generate one batch of data"""
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]

        batch_size = len(batch_indexes)
        batch_x = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
        batch_y_coords = np.zeros((batch_size, 2), dtype=np.float32)
        batch_y_cities = np.zeros((batch_size, len(self.city_encoder.classes_)), dtype=np.float32)

        for i, (_, row) in enumerate(batch_metadata.iterrows()):
            img_path = self.images_dir / row['filename']
            img = preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = preprocessing.image.img_to_array(img)

            if self.augment:
                img = self.aug_layer(img)

            img = applications.efficientnet.preprocess_input(img)

            batch_x[i] = img
            batch_y_coords[i] = self.normalize_coordinates(row['latitude'], row['longitude'])
            batch_y_cities[i] = utils.to_categorical(row['city_encoded'], num_classes=len(self.city_encoder.classes_))

        return batch_x, {
            'city': batch_y_cities,
            'coordinates': batch_y_coords
        }

    def on_epoch_end(self):
        """Shuffles the data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)