import pandas as pd
import numpy as np
from pathlib import Path
from keras import Sequential, utils, layers, preprocessing, applications

class DataGenerator(utils.Sequence):
    def __init__(self, metadata_path: Path, images_dir: Path, batch_size: int = 32, augment: bool = False, shuffle: bool = True):
        self.metadata = pd.read_csv(metadata_path)
        self.images_dir = images_dir
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.metadata))

        if self.augment:
            self.aug_layer = Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomBrightness(0.2),
                layers.RandomContrast(0.2)
            ])

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.metadata) / self.batch_size))

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_metadata = self.metadata.iloc[batch_indexes]

        batch_x = np.zeros((len(batch_indexes), 224, 224, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_indexes), 2), dtype=np.float32)

        for i, (_, row) in enumerate(batch_metadata.iterrows()):
            img_path = self.images_dir / f"processed_{row['filename']}"
            img = preprocessing.image.load_img(img_path, target_size=(224, 224))
            img = preprocessing.image.img_to_array(img)

            if self.augment:
                img = self.aug_layer(img)

            img = applications.resnet50.preprocess_input(img)

            batch_x[i] = img
            batch_y[i] = [row['latitude'], row['longitude']]

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)