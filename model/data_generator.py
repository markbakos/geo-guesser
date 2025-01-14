import pandas as pd
import numpy as np
from pathlib import Path

from albumentations.core.utils import LabelEncoder
from keras import Sequential, utils, layers, preprocessing, applications

class DataGenerator(utils.Sequence):
    def __init__(self, metadata_path: Path, images_dir: Path, batch_size: int = 32, augment: bool = False, shuffle: bool = True):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))

        self.create_region_labels()

        if self.augment:
            self.aug_layer = Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2)
            ])

        self.on_epoch_end()

    def create_region_labels(self):
        from sklearn.cluster import KMeans

        coords = self.metadata[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=8, random_state=42)
        self.metadata['region'] = kmeans.fit_predict(coords)

        self.label_encoder = LabelEncoder()
        self.metadata['region_encoded'] = self.label_encoder.fit_transform(self.metadata['region'])

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]

        batch_x = np.zeros((len(batch_indexes), 224, 224, 3), dtype=np.float32)
        batch_y_coords = np.zeros((len(batch_indexes), 2), dtype=np.float32)
        batch_y_regions = np.zeros((len(batch_indexes), 8), dtype=np.float32)

        for i, (_, row) in enumerate(batch_metadata.iterrows()):
            img_path = self.images_dir / f"processed_{row['filename']}"
            img = preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = preprocessing.image.img_to_array(img)

            if self.augment:
                img = self.aug_layer(img)

            img = applications.resnet50.preprocess_input(img)

            batch_x[i] = img
            batch_y_coords[i] = [row['latitude'], row['longitude']]
            batch_y_regions[i] = utils.to_categorical(row['region_encoded'], num_classes=8)

        return batch_x, {'region': batch_y_regions, 'coordinates': batch_y_coords}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)