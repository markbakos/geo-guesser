import tensorflow as tf
from pathlib import Path
import numpy as np
from keras import callbacks, optimizers, preprocessing, applications
from .data_generator import DataGenerator
from .metrics import haversine_loss, location_accuracy
from .model import create_model

class LocationTrainer:
    def __init__(self, data_dir: str = "collection/dataset", batch_size: int = 16, epochs: int = 200, initial_lr: float = 0.0005):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.model = None
        self.checkpoint_dir = Path("model/checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self.best_model_dir = self.checkpoint_dir / "best_model"
        self.best_model_dir.mkdir(exist_ok=True)

        self.backup_dir = self.checkpoint_dir / "backup"
        self.backup_dir.mkdir(exist_ok=True)

    def prepare_generators(self):
        train_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/train_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=True,
        )

        val_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/val_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=False,
        )

        test_gen = DataGenerator(
            metadata_path=self.data_dir / "metadata/test_metadata.csv",
            images_dir=self.data_dir / "images",
            batch_size=self.batch_size,
            augment=False,
        )

        return train_gen, val_gen, test_gen

    def get_callbacks(self):
        callback = [
            callbacks.ModelCheckpoint(
                str(self.best_model_dir / "best_location_model.keras"),
                monitor='val_coordinates_location_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                str(self.best_model_dir / "best_overall_model.keras"),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_coordinates_location_accuracy',
                mode='max',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_coordinates_loss',
                mode='min',
                factor=0.2,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.CSVLogger(
                str(self.checkpoint_dir / "training_log.csv"),
                separator=',',
                append=True
            ),
            callbacks.BackupAndRestore(
                backup_dir=str(self.checkpoint_dir / "backup")
            )
        ]

        return callback

    def train(self):
        train_gen, val_gen, test_gen = self.prepare_generators()

        self.model = create_model()

        optimizer = optimizers.Adam(learning_rate=self.initial_lr)

        self.model.compile(
            optimizer=optimizer,
            loss={
                'region': 'categorical_crossentropy',
                'coordinates': haversine_loss,
            },
            loss_weights = {
                'region': 0.3,
                'coordinates': 0.7,
            },
            metrics={
                'region': 'accuracy',
                'coordinates': [location_accuracy],
            }
        )

        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.epochs,
            callbacks=self.get_callbacks(),
            shuffle=True,
        )

        test_results = self.model.evaluate(test_gen)
        print("Test results:")
        for metric_name, value in zip(self.model.metrics_names, test_results):
            print(f"{metric_name}: {value:.4f}")

        return history

    def predict_location(self, image_path: str) -> dict:
        if self.model is None:
            raise ValueError("Model needs to be trained first")

        img = preprocessing.image.load_img(
            image_path, target_size=(224, 224)
        )

        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.efficientnet.preprocess_input(img_array)

        region_probs, coordinates = self.model.predict(img_array)
        predicted_region = np.argmax(region_probs[0])

        return {
            'coordinates': tuple(coordinates[0]),
            'region': predicted_region,
            'region_confidence': float(region_probs[0][predicted_region])
        }