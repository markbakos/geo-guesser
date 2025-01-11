import logging
from importlib.metadata import metadata
from pathlib import Path
from PIL import Image
from typing import Tuple, Dict
import pandas as pd

class GeolocationDataCollector:
    def __init__(self, base_path: str = "dataset"):
        self.base_path = Path(base_path)
        self.images_path = self.base_path / "images"
        self.metadata_path = self.base_path / "metadata"

        self.images_path.mkdir(parents=True, exist_ok=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.base_path / 'data_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_image(self, image_path: str) -> bool:
        """Validates if image is usable, not corrupted and has proper dimensions"""

        try:
            with Image.open(image_path) as img:
                img.verify()
                img = Image.open(image_path)

                if img.size[0] < 224 or img.size[1] < 224:
                    return False
                return True
        except Exception as e:
            self.logger.error(f"Error validating image {image_path}: {str(e)}")
            return False

    def process_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> str:
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img.thumbnail(target_size, Image.Resampling.LANCZOS)

                new_img = Image.new('RGB', target_size, (0, 0, 0))
                new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))

                processed_path = str(Path(image_path).parent / f"processed_{Path(image_path).name}")
                new_img.save(processed_path, 'JPEG', quality=95)
                return processed_path
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {str(e)}")
            return ""

    def save_metadata(self, metadata: pd.DataFrame, filename: str = "metadata.csv") -> None:
        try:
            metadata_file = self.metadata_path / filename
            metadata.to_csv(metadata_file, index=False)
            self.logger.info(f"Metadata saved to {metadata_file}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")

    def create_dataset_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Splits dataset into training validation and test"""

        try:
            metadata = pd.read_csv(self.metadata_path / "metadata.csv")

            metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)

            train_size = int(len(metadata) * train_ratio)
            val_size = int(len(metadata) * val_ratio)

            train_data = metadata[:train_size]
            val_data = metadata[train_size:train_size + val_size]
            test_data = metadata[train_size + val_size:]

            splits = {
                'train': train_data,
                'val': val_data,
                'test': test_data,
            }

            for split_name, split_data in splits.items():
                split_data.to_csv(self.metadata_path / f"{split_name}_metadata.csv", index=False)

            return splits

        except Exception as e:
            self.logger.error(f"Error creating dataset split: {str(e)}")
            return {}
