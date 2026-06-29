import os
import sys
import argparse

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, PROJECT_ROOT)

from src.configs import ExperimentConfig
from src.experiments import run_baseline


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--chexpert_root", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--valid_csv", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig()
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.learning_rate = args.learning_rate

    history = run_baseline(
        cfg=cfg,
        chexpert_root=args.chexpert_root,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
    )

    print(history)


if __name__ == "__main__":
    main()