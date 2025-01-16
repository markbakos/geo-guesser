import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

class DatasetSplitter:
    def __init__(self, metadata_path: str, output_dir: str = "dataset/metadata"):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=self.output_dir.parent / 'split_log.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42) -> dict:
        try:
            df = pd.read_csv(self.metadata_path)
            self.logger.info(f"Loaded {len(df)} samples from {self.metadata_path}")

            train_df, temp_df = train_test_split(
                df,
                train_size=train_ratio,
                stratify=df['region'],
                random_state=seed
            )

            val_ratio_adjusted = val_ratio / (1 - train_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                stratify=temp_df['region'],
                random_state=seed
            )

            splits = {
                'train': train_df,
                'val': val_df,
                'test': test_df
            }

            for split_name, split_df in splits.items():
                output_path = self.output_dir / f"{split_name}_metadata.csv"
                split_df.to_csv(output_path, index=False)
                self.logger.info(f"Saved {split_name} split with {len(split_df)} samples to {output_path}")

            return splits

        except Exception as e:
            self.logger.error(f"Error during dataset splitting: {str(e)}")
            raise


def main():
    splitter = DatasetSplitter(
        metadata_path="dataset/metadata/flickr_metadata.csv"
    )

    splits = splitter.split_dataset()

    print("\nDataset split complete!")
    print("\nSplit sizes:")
    for split_name, split_df in splits.items():
        print(f"{split_name}: {len(split_df)} samples")

    print("\nLocation type distribution in splits:")
    for split_name, split_df in splits.items():
        print(f"\n{split_name} split distribution:")
        print(split_df['region'].value_counts(normalize=True).mul(100).round(1))


if __name__ == "__main__":
    main()