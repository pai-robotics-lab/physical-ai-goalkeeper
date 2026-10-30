"""Learning and evaluation utilities for the AI goalkeeper experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class TrajectoryConfig:
    """Parameters shared by trajectory generation, training, and evaluation."""

    dt: float = 0.02
    start_x: float = 2.5
    goal_x: float = 0.0
    observation_steps: tuple[int, ...] = (0, 10, 20, 30)
    max_steps: int = 300
    sensor_noise_std: float = 0.004
    goalie_half_width: float = 0.18
    goalie_max_speed: float = 2.5


@dataclass(frozen=True)
class Shot:
    """Initial conditions and hidden dynamics for one simulated shot."""

    initial_vx: float
    initial_vy: float
    drag: float
    spin: float


@dataclass(frozen=True)
class Trajectory:
    """One complete ball trajectory and the observations available to the controller."""

    positions: np.ndarray
    observation: np.ndarray
    crossing_y: float
    crossing_step: int


def sample_shot(rng: np.random.Generator) -> Shot:
    """Draw one shot from the experiment's reproducible parameter distribution."""

    return Shot(
        initial_vx=float(rng.uniform(-2.2, -1.15)),
        initial_vy=float(rng.uniform(-0.8, 0.8)),
        drag=float(rng.uniform(0.03, 0.16)),
        spin=float(rng.uniform(-0.75, 0.75)),
    )


def simulate_shot(
    shot: Shot,
    config: TrajectoryConfig,
    *,
    rng: np.random.Generator | None = None,
) -> Trajectory:
    """Integrate a nonlinear rolling-ball model until it crosses the goal line."""

    position = np.array([config.start_x, 0.0], dtype=np.float64)
    velocity = np.array([shot.initial_vx, shot.initial_vy], dtype=np.float64)
    positions = [position.copy()]

    crossing_y: float | None = None
    crossing_step = -1
    for step in range(1, config.max_steps + 1):
        previous = position.copy()
        speed = float(np.linalg.norm(velocity))
        acceleration = -shot.drag * speed * velocity
        acceleration[1] += shot.spin * abs(velocity[0])
        velocity += acceleration * config.dt
        position += velocity * config.dt
        positions.append(position.copy())

        if position[0] <= config.goal_x:
            fraction = (previous[0] - config.goal_x) / (previous[0] - position[0])
            crossing_y = float(previous[1] + fraction * (position[1] - previous[1]))
            crossing_step = step
            break

    if crossing_y is None:
        raise RuntimeError("Shot did not reach the goal line within max_steps")

    positions_array = np.asarray(positions, dtype=np.float32)
    observation = positions_array[list(config.observation_steps)].reshape(-1).copy()
    if rng is not None and config.sensor_noise_std > 0:
        observation += rng.normal(0.0, config.sensor_noise_std, size=observation.shape)

    return Trajectory(
        positions=positions_array,
        observation=observation.astype(np.float32),
        crossing_y=crossing_y,
        crossing_step=crossing_step,
    )


def generate_dataset(
    sample_count: int,
    *,
    seed: int,
    config: TrajectoryConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy observations and exact goal-line crossing targets."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    config = config or TrajectoryConfig()
    rng = np.random.default_rng(seed)
    features = np.empty((sample_count, len(config.observation_steps) * 2), dtype=np.float32)
    targets = np.empty((sample_count, 1), dtype=np.float32)
    for index in range(sample_count):
        trajectory = simulate_shot(sample_shot(rng), config, rng=rng)
        features[index] = trajectory.observation
        targets[index, 0] = trajectory.crossing_y
    return features, targets


def linear_prediction(observation: np.ndarray, goal_x: float = 0.0) -> np.ndarray:
    """Predict the crossing point by extending the last observed displacement."""

    values = np.asarray(observation, dtype=np.float32)
    original_shape = values.shape
    rows = values.reshape(-1, original_shape[-1])
    previous = rows[:, -4:-2]
    current = rows[:, -2:]
    delta = current - previous
    safe_dx = np.where(np.abs(delta[:, 0]) < 1e-6, -1e-6, delta[:, 0])
    steps = (goal_x - current[:, 0]) / safe_dx
    prediction = current[:, 1] + steps * delta[:, 1]
    return prediction.reshape(original_shape[:-1] + (1,))


class TrajectoryRegressor(nn.Module):
    """Small normalized MLP that predicts the ball's goal-line crossing position."""

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_std", input_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.network = nn.Sequential(
            nn.Linear(input_mean.numel(), 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.input_mean) / self.input_std
        prediction = self.network(normalized)
        return prediction * self.target_std + self.target_mean


@dataclass(frozen=True)
class TrainingResult:
    """Trained model plus compact metrics needed by the report."""

    model: TrajectoryRegressor
    metrics: dict[str, float]


def train_model(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int = 7,
    epochs: int = 300,
    learning_rate: float = 0.01,
) -> TrainingResult:
    """Train the deterministic CPU MLP and compare it with linear extrapolation."""

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    features = torch.from_numpy(train_features)
    targets = torch.from_numpy(train_targets)
    validation_x = torch.from_numpy(validation_features)

    input_std = features.std(dim=0).clamp_min(1e-6)
    target_std = targets.std(dim=0).clamp_min(1e-6)
    model = TrajectoryRegressor(
        features.mean(dim=0),
        input_std,
        targets.mean(dim=0),
        target_std,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        prediction = model(validation_x).numpy()
    baseline = linear_prediction(validation_features)
    metrics = evaluate_predictions(prediction, baseline, validation_targets)
    metrics["train_loss"] = float(loss.detach())
    return TrainingResult(model=model, metrics=metrics)


def evaluate_predictions(
    ai_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    targets: np.ndarray,
    *,
    goalie_half_width: float = 0.18,
) -> dict[str, float]:
    """Compute prediction error and idealized interception success rates."""

    ai_error = np.abs(ai_prediction - targets)
    baseline_error = np.abs(baseline_prediction - targets)
    return {
        "ai_mae": float(ai_error.mean()),
        "baseline_mae": float(baseline_error.mean()),
        "ai_save_rate": float((ai_error <= goalie_half_width).mean()),
        "baseline_save_rate": float((baseline_error <= goalie_half_width).mean()),
    }


def checkpoint_payload(
    model: TrajectoryRegressor,
    config: TrajectoryConfig,
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Return a portable, metadata-rich model checkpoint."""

    return {
        "state_dict": model.state_dict(),
        "input_mean": model.input_mean,
        "input_std": model.input_std,
        "target_mean": model.target_mean,
        "target_std": model.target_std,
        "config": asdict(config),
        "metrics": metrics,
    }


def load_checkpoint(path: str) -> tuple[TrajectoryRegressor, TrajectoryConfig, dict[str, float]]:
    """Load a checkpoint created by :func:`checkpoint_payload`."""

    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = TrajectoryRegressor(
        payload["input_mean"],
        payload["input_std"],
        payload["target_mean"],
        payload["target_std"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    config = TrajectoryConfig(**payload["config"])
    metrics = {key: float(value) for key, value in payload["metrics"].items()}
    return model, config, metrics
