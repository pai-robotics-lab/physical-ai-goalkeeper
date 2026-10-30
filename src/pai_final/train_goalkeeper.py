"""Train and evaluate the AI goalkeeper trajectory predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .goalkeeper import TrajectoryConfig, checkpoint_payload, generate_dataset, train_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AI goalkeeper predictor")
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "goalkeeper_model.pt",
    )
    args = parser.parse_args()

    config = TrajectoryConfig()
    train_features, train_targets = generate_dataset(
        args.train_samples,
        seed=args.seed,
        config=config,
    )
    validation_features, validation_targets = generate_dataset(
        args.validation_samples,
        seed=args.seed + 1,
        config=config,
    )
    result = train_model(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        seed=args.seed,
        epochs=args.epochs,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(result.model, config, result.metrics), args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Model: {args.output}")
    print(f"Metrics: {metrics_path}")
    for name, value in result.metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
