import argparse
from pathlib import Path
from .train import LocationTrainer


def main():
    parser = argparse.ArgumentParser(description='Train geolocation model')
    parser.add_argument('--data_dir', type=str, default='collection/dataset',
                        help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')

    args = parser.parse_args()

    trainer = LocationTrainer(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        initial_lr=args.lr
    )

    history = trainer.train()
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()