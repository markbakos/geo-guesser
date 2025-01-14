import tensorflow as tf
from pathlib import Path
import numpy as np
from keras import callbacks, optimizers, preprocessing, applications
from .data_generator import DataGenerator
from .architecture import create_model
from .metrics import haversine_loss, location_accuracy

class LocationTrainer:
    def __init__(self, data_dir: str = "collection/dataset", batch_size: int = 32, epochs: int = 100, initial_lr: float = 0.001):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.model = None
        self.checkpoint_dir = Path("model/checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

    def prepare_generators(self):
        train_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/train_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=True,
        )

        val_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/train_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=False,
        )

        test_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/train_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=False,
        )

        return train_gen, val_gen, test_gen

    def get_callbacks(self):
        callback = [
            callbacks.ModelCheckpoint(
                str(self.checkpoint_dir / "best_model.keras"),
                monitor='val_location_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_location_accuracy',
                mode='max',
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        return callback

    def train(self):
        self.model = create_model()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.initial_lr),
            loss=haversine_loss,
            metrics=[location_accuracy]
        )

        train_gen, val_gen, test_gen = self.prepare_generators()

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=self.get_callbacks(),
        )

        test_results = self.model.evaluate(test_gen)
        print(f"\nTest Loss: {test_results[0]:.4f}")
        print(f"Test Accuracy: {test_results[1]:.4f}")

        return history

    def predict_location(self, image_path: str) -> tuple:
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        img = preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )

        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.resnet50.preprocess_input(img_array)

        coordinates = self.model.predict(img_array)[0]
        return tuple[coordinates]
