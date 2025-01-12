import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "collection"))

from collection.data_collection import GeolocationDataCollector


def prepare_dataset():
    """Prepare the dataset by creating train/val/test splits"""
    print("Creating dataset splits...")

    base_path = Path(__file__).parent.parent / "collection" / "dataset"
    collector = GeolocationDataCollector(str(base_path))

    splits = collector.create_dataset_split(train_ratio=0.8, val_ratio=0.1)

    if splits:
        print("\nDataset split successfully created:")
        for split_name, split_data in splits.items():
            print(f"{split_name}: {len(split_data)} samples")
    else:
        print("Error creating dataset splits!")


if __name__ == "__main__":
    prepare_dataset()