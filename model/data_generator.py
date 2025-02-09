import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras import Sequential, utils, layers, preprocessing, applications
import pickle
from sklearn.cluster import KMeans


class DataGenerator(utils.Sequence):
    def __init__(self, metadata_path: Path, images_dir: Path, batch_size: int = 32, augment: bool = False, shuffle: bool = True, n_clusters: int = 6):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))
        self.n_clusters = n_clusters

        if 'region' not in self.metadata.columns:
            self.metadata['region'] = 'unknown'
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
        coords = self.metadata[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        self.metadata['region'] = kmeans.fit_predict(coords)

        self.region_encoder = LabelEncoder()
        self.metadata['region_encoded'] = self.region_encoder.fit_transform(self.metadata['region'])

        encoder_path = Path("model/checkpoints/region_encoder.pkl")
        encoder_path.parent.mkdir(exist_ok=True, parents=True)
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.region_encoder, f)

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]

        batch_size = len(batch_indexes)
        batch_x = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
        batch_y_coords = np.zeros((batch_size, 2), dtype=np.float32)
        batch_y_regions = np.zeros((batch_size, self.n_clusters), dtype=np.float32)

        for i, (_, row) in enumerate(batch_metadata.iterrows()):
            img_path = self.images_dir / f"{row['filename']}"
            img = preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = preprocessing.image.img_to_array(img)

            if self.augment:
                img = self.aug_layer(img)

            img = applications.efficientnet.preprocess_input(img)

            batch_x[i] = img
            batch_y_coords[i] = [row['latitude'], row['longitude']]
            batch_y_regions[i] = utils.to_categorical(row['region_encoded'], num_classes=self.n_clusters)

        return batch_x, {
            'region': batch_y_regions,
            'coordinates': batch_y_coords
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)