"""Reproduce and document the frozen AI goalkeeper submission results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

_matplotlib_cache = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib
import numpy as np
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans"})

from pai_final.goalkeeper import (
    TrajectoryConfig,
    TrajectoryRegressor,
    evaluate_predictions,
    generate_dataset,
    linear_prediction,
    load_checkpoint,
    sample_shot,
    simulate_shot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "validation"
FROZEN_MANIFEST = PROJECT_ROOT / "validation" / "goalkeeper_frozen_artifacts.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_frozen_artifacts() -> list[dict[str, object]]:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    for item in manifest["files"]:
        path = PROJECT_ROOT / item["path"]
        actual_hash = sha256(path) if path.exists() else None
        actual_bytes = path.stat().st_size if path.exists() else None
        passed = actual_hash == item["sha256"] and actual_bytes == item["bytes"]
        checks.append(
            {
                "path": item["path"],
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
                "expected_bytes": item["bytes"],
                "actual_bytes": actual_bytes,
                "passed": passed,
            }
        )
    if not all(bool(check["passed"]) for check in checks):
        raise RuntimeError("One or more frozen artifacts changed; refusing to continue")
    return checks


def model_architecture(model: TrajectoryRegressor) -> list[str]:
    return [type(layer).__name__ for layer in model.network]


def reproduce_training(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    reference_model: TrajectoryRegressor,
    *,
    seed: int,
    epochs: int,
) -> tuple[list[float], float]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    features = torch.from_numpy(train_features)
    targets = torch.from_numpy(train_targets)
    model = TrajectoryRegressor(
        features.mean(dim=0),
        features.std(dim=0).clamp_min(1e-6),
        targets.mean(dim=0),
        targets.std(dim=0).clamp_min(1e-6),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    max_difference = 0.0
    for name, reference in reference_model.state_dict().items():
        candidate = model.state_dict()[name]
        max_difference = max(max_difference, float(torch.max(torch.abs(reference - candidate))))
    return losses, max_difference


def predict(model: TrajectoryRegressor, observation: np.ndarray, controller: str) -> float:
    if controller == "baseline":
        return float(linear_prediction(observation)[0])
    with torch.inference_mode():
        return float(model(torch.from_numpy(observation).unsqueeze(0)).squeeze())


def build_robot_scene(config: TrajectoryConfig):
    cache_path = PROJECT_ROOT / ".cache" / "quadrants"
    os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(cache_path))
    os.environ.setdefault(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / "gstaichi-cpu"),
    )
    import genesis as gs

    gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=(0.0, 0.0, 0.0)),
        show_viewer=False,
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(pos=(config.start_x, 0.0, 0.10), radius=0.08, collision=False),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(PROJECT_ROOT / "assets" / "goalkeeper_robot.urdf"),
            pos=(0.14, 0.0, 0.0),
            fixed=True,
            prioritize_urdf_material=True,
        )
    )
    scene.build()
    robot.set_dofs_kp([2200.0])
    robot.set_dofs_kv([95.0])
    robot.set_dofs_force_range([-800.0], [800.0])
    return scene, ball, robot, getattr(gs, "__version__", "unknown")


def validate_robot_control(
    model: TrajectoryRegressor,
    config: TrajectoryConfig,
    *,
    shots: int,
    seed: int,
) -> tuple[list[dict[str, object]], str]:
    scene, ball, robot, genesis_version = build_robot_scene(config)
    rows: list[dict[str, object]] = []
    observation_end = max(config.observation_steps)
    for controller in ("ai", "baseline"):
        rng = np.random.default_rng(seed)
        for shot_index in range(shots):
            trajectory = simulate_shot(sample_shot(rng), config, rng=rng)
            predicted_y = predict(model, trajectory.observation, controller)
            controller_target = 0.0
            robot.set_dofs_position([0.0], zero_velocity=True)
            robot.control_dofs_position([0.0])
            for step, position in enumerate(trajectory.positions):
                ball.set_pos(
                    (float(position[0]), float(position[1]), 0.10),
                    zero_velocity=True,
                )
                if step >= observation_end:
                    delta = np.clip(
                        predicted_y - controller_target,
                        -config.goalie_max_speed * config.dt,
                        config.goalie_max_speed * config.dt,
                    )
                    controller_target += float(delta)
                controller_target = float(np.clip(controller_target, -2.35, 2.35))
                robot.control_dofs_position([controller_target])
                scene.step()

            actual_robot_y = float(robot.get_dofs_position()[0])
            error = abs(actual_robot_y - trajectory.crossing_y)
            rows.append(
                {
                    "controller": controller,
                    "shot": shot_index + 1,
                    "predicted_y": predicted_y,
                    "commanded_joint_y": controller_target,
                    "actual_robot_y": actual_robot_y,
                    "ball_crossing_y": trajectory.crossing_y,
                    "robot_tracking_error": abs(actual_robot_y - controller_target),
                    "interception_error": error,
                    "saved": error <= config.goalie_half_width,
                }
            )
    return rows, genesis_version


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curve(path: Path, losses: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(np.arange(1, len(losses) + 1), losses, color="#2478C8", linewidth=2)
    axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training MSE (log scale)")
    axis.set_title("Reproduced training curve (seed 7)")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_predictions(
    path: Path,
    targets: np.ndarray,
    ai_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
) -> None:
    values = np.concatenate([targets.ravel(), ai_prediction.ravel(), baseline_prediction.ravel()])
    lower, upper = float(values.min()), float(values.max())
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharex=True, sharey=True)
    for axis, title, prediction, color in (
        (axes[0], "AI (MLP)", ai_prediction, "#1A9B72"),
        (axes[1], "Constant-velocity baseline", baseline_prediction, "#E68A17"),
    ):
        axis.scatter(targets, prediction, s=9, alpha=0.35, color=color, edgecolors="none")
        axis.plot([lower, upper], [lower, upper], color="#435466", linewidth=1.2)
        axis.set_title(title)
        axis.set_xlabel("Actual crossing y (m)")
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("Predicted crossing y (m)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_success_rates(
    path: Path,
    proxy_metrics: dict[str, float],
    robot_rows: list[dict[str, object]],
) -> None:
    robot_rates = {
        controller: np.mean(
            [bool(row["saved"]) for row in robot_rows if row["controller"] == controller]
        )
        for controller in ("ai", "baseline")
    }
    labels = ["AI\nproxy n=1000", "Baseline\nproxy n=1000", "AI\nGenesis n=10", "Baseline\nGenesis n=10"]
    values = [
        proxy_metrics["ai_save_rate"],
        proxy_metrics["baseline_save_rate"],
        robot_rates["ai"],
        robot_rates["baseline"],
    ]
    colors_list = ["#1A9B72", "#E68A17", "#1A9B72", "#E68A17"]
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    bars = axis.bar(labels, np.asarray(values) * 100.0, color=colors_list)
    axis.set_ylim(0, 108)
    axis.set_ylabel("Interception success rate (%)")
    axis.set_title("Prediction proxy and measured Genesis robot results")
    axis.bar_label(bars, fmt="%.1f%%", padding=3)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen goalkeeper submission")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-shots", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frozen_checks = verify_frozen_artifacts()
    model_path = PROJECT_ROOT / "artifacts" / "goalkeeper_model.pt"
    model, config, checkpoint_metrics = load_checkpoint(str(model_path))

    train_features, train_targets = generate_dataset(4000, seed=7, config=config)
    validation_features, validation_targets = generate_dataset(1000, seed=8, config=config)
    with torch.inference_mode():
        ai_prediction = model(torch.from_numpy(validation_features)).numpy()
    baseline_prediction = linear_prediction(validation_features)
    reproduced_metrics = evaluate_predictions(
        ai_prediction,
        baseline_prediction,
        validation_targets,
        goalie_half_width=config.goalie_half_width,
    )
    metric_delta = {
        key: abs(reproduced_metrics[key] - checkpoint_metrics[key])
        for key in reproduced_metrics
    }
    if max(metric_delta.values()) > 1e-8:
        raise RuntimeError(f"Checkpoint metrics did not reproduce: {metric_delta}")

    losses, max_weight_difference = reproduce_training(
        train_features,
        train_targets,
        model,
        seed=7,
        epochs=300,
    )
    if max_weight_difference > 1e-8:
        raise RuntimeError(f"Reproduced model weights differ: {max_weight_difference}")

    prediction_rows = [
        {
            "sample": index + 1,
            "actual_crossing_y": float(validation_targets[index, 0]),
            "ai_prediction_y": float(ai_prediction[index, 0]),
            "baseline_prediction_y": float(baseline_prediction[index, 0]),
            "ai_abs_error": float(abs(ai_prediction[index, 0] - validation_targets[index, 0])),
            "baseline_abs_error": float(
                abs(baseline_prediction[index, 0] - validation_targets[index, 0])
            ),
        }
        for index in range(len(validation_targets))
    ]
    robot_rows, genesis_version = validate_robot_control(
        model,
        config,
        shots=args.robot_shots,
        seed=21,
    )
    robot_summary: dict[str, dict[str, float]] = {}
    for controller in ("ai", "baseline"):
        selected = [row for row in robot_rows if row["controller"] == controller]
        robot_summary[controller] = {
            "shots": len(selected),
            "save_rate": float(np.mean([bool(row["saved"]) for row in selected])),
            "mean_interception_error": float(
                np.mean([float(row["interception_error"]) for row in selected])
            ),
            "max_robot_tracking_error": float(
                np.max([float(row["robot_tracking_error"]) for row in selected])
            ),
        }

    curve_rows = [
        {"epoch": index + 1, "training_mse": loss} for index, loss in enumerate(losses)
    ]
    write_csv(args.output / "training_curve.csv", curve_rows)
    write_csv(args.output / "predictions.csv", prediction_rows)
    write_csv(args.output / "robot_trials.csv", robot_rows)
    plot_training_curve(args.output / "training_curve.png", losses)
    plot_predictions(
        args.output / "prediction_scatter.png",
        validation_targets,
        ai_prediction,
        baseline_prediction,
    )
    plot_success_rates(args.output / "success_rates.png", reproduced_metrics, robot_rows)

    result = {
        "frozen_artifacts": frozen_checks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "genesis": genesis_version,
            "backend": "cpu",
        },
        "model": {
            "class": type(model).__name__,
            "layers": model_architecture(model),
            "config": asdict(config),
            "retrained_max_weight_difference": max_weight_difference,
            "mlp_inference_used": True,
        },
        "validation": {
            "train_samples": 4000,
            "validation_samples": 1000,
            "train_seed": 7,
            "validation_seed": 8,
            "epochs": 300,
            "checkpoint_metrics": checkpoint_metrics,
            "reproduced_metrics": reproduced_metrics,
            "metric_delta": metric_delta,
            "success_rate_definition": (
                "prediction proxy: abs(predicted crossing y - actual crossing y) <= 0.18 m"
            ),
        },
        "genesis_robot": {
            "seed": 21,
            "joint": "goalkeeper_slide",
            "joint_type": "prismatic",
            "controller": "PD position control",
            "success_definition": (
                "abs(actual Genesis joint position - ball crossing y) <= 0.18 m"
            ),
            "results": robot_summary,
            "actual_joint_position_used": True,
        },
    }
    (args.output / "goalkeeper_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["validation"]["reproduced_metrics"], indent=2))
    print(json.dumps(result["genesis_robot"]["results"], indent=2))
    print(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
