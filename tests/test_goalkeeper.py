from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from pai_final.goalkeeper import (
    TrajectoryConfig,
    generate_dataset,
    linear_prediction,
    train_model,
)


def test_dataset_is_reproducible() -> None:
    first = generate_dataset(8, seed=123)
    second = generate_dataset(8, seed=123)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_linear_prediction_reaches_goal_line() -> None:
    observation = np.array([[2.5, 0.0, 2.0, 0.1, 1.5, 0.2, 1.0, 0.3]], dtype=np.float32)

    prediction = linear_prediction(observation)

    np.testing.assert_allclose(prediction, [[0.5]], atol=1e-6)


def test_training_improves_on_nonlinear_trajectory() -> None:
    config = TrajectoryConfig(sensor_noise_std=0.0)
    train_features, train_targets = generate_dataset(600, seed=4, config=config)
    validation_features, validation_targets = generate_dataset(200, seed=5, config=config)

    result = train_model(
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        epochs=100,
    )

    assert result.metrics["ai_mae"] < result.metrics["baseline_mae"] * 0.6
    assert result.metrics["ai_save_rate"] >= 0.9


def test_robot_urdf_defines_prismatic_actuator() -> None:
    project_root = Path(__file__).resolve().parents[1]
    root = ElementTree.parse(project_root / "assets" / "goalkeeper_robot.urdf").getroot()

    joint = root.find("./joint[@name='goalkeeper_slide']")

    assert joint is not None
    assert joint.attrib["type"] == "prismatic"
    assert joint.find("axis").attrib["xyz"] == "0 1 0"
