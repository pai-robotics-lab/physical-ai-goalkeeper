"""Run the trained AI goalkeeper in a Genesis scene."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from .goalkeeper import linear_prediction, load_checkpoint, sample_shot, simulate_shot

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _predict(model: torch.nn.Module, observation: np.ndarray, controller: str) -> float:
    if controller == "baseline":
        return float(linear_prediction(observation)[0])
    with torch.inference_mode():
        features = torch.from_numpy(observation).unsqueeze(0)
        return float(model(features).squeeze())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Genesis AI goalkeeper demo")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "goalkeeper_model.pt",
    )
    parser.add_argument("--controller", choices=("ai", "baseline"), default="ai")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--show-viewer", action="store_true")
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()

    if args.shots < 1:
        parser.error("--shots must be positive")
    if not args.model.exists():
        parser.error(f"model not found: {args.model}; run pai_final.train_goalkeeper first")

    cache_name = "quadrants" if args.backend == "cpu" else f"quadrants-{args.backend}"
    os.environ.setdefault("QD_OFFLINE_CACHE_FILE_PATH", str(PROJECT_ROOT / ".cache" / cache_name))
    os.environ.setdefault(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / f"gstaichi-{args.backend}"),
    )

    import genesis as gs

    model, config, training_metrics = load_checkpoint(str(args.model))
    backend = getattr(gs, args.backend, None)
    if backend is None:
        raise RuntimeError(f"Genesis does not support backend {args.backend!r}")
    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=config.dt, gravity=(0.0, 0.0, 0.0)),
        show_viewer=args.show_viewer,
    )
    scene.add_entity(
        gs.morphs.Box(pos=(1.25, 0.0, -0.025), size=(2.8, 6.0, 0.05), fixed=True),
        surface=gs.surfaces.Default(color=(0.15, 0.35, 0.18, 1.0)),
    )
    ball = scene.add_entity(
        gs.morphs.Sphere(pos=(config.start_x, 0.0, 0.10), radius=0.08, collision=False),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.10, 1.0)),
    )
    goalie = scene.add_entity(
        gs.morphs.URDF(
            file=str(PROJECT_ROOT / "assets" / "goalkeeper_robot.urdf"),
            pos=(0.14, 0.0, 0.0),
            fixed=True,
            prioritize_urdf_material=True,
        ),
    )
    marker = scene.add_entity(
        gs.morphs.Box(pos=(-0.02, 0.0, 0.02), size=(0.04, 0.16, 0.04), collision=False),
        surface=gs.surfaces.Default(color=(0.95, 0.18, 0.18, 1.0)),
    )
    for side in (-2.55, 2.55):
        scene.add_entity(
            gs.morphs.Box(pos=(0.0, side, 0.35), size=(0.08, 0.08, 0.7), fixed=True),
            surface=gs.surfaces.Default(color=(0.9, 0.9, 0.9, 1.0)),
        )

    camera = None
    if args.record is not None:
        camera = scene.add_camera(
            res=(960, 540),
            pos=(4.7, -4.8, 5.8),
            lookat=(1.1, 0.0, 0.0),
            fov=42,
            GUI=False,
        )
    scene.build()
    goalie.set_dofs_kp([2200.0])
    goalie.set_dofs_kv([95.0])
    goalie.set_dofs_force_range([-800.0], [800.0])
    if camera is not None:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        camera.start_recording()

    rng = np.random.default_rng(args.seed)
    saves = 0
    absolute_errors: list[float] = []
    observation_end = max(config.observation_steps)
    for shot_index in range(args.shots):
        trajectory = simulate_shot(sample_shot(rng), config, rng=rng)
        predicted_y = _predict(model, trajectory.observation, args.controller)
        controller_target = 0.0
        goalie.set_dofs_position([0.0], zero_velocity=True)
        goalie.control_dofs_position([0.0])
        marker.set_pos((-0.02, predicted_y, 0.02), zero_velocity=True)

        for step, position in enumerate(trajectory.positions):
            ball.set_pos((float(position[0]), float(position[1]), 0.10), zero_velocity=True)
            if step >= observation_end:
                delta = np.clip(
                    predicted_y - controller_target,
                    -config.goalie_max_speed * config.dt,
                    config.goalie_max_speed * config.dt,
                )
                controller_target += float(delta)
            controller_target = float(np.clip(controller_target, -2.35, 2.35))
            goalie.control_dofs_position([controller_target])
            scene.step()
            if camera is not None and step % 2 == 0:
                camera.render()

        goalie_y = float(goalie.get_dofs_position()[0])
        error = abs(goalie_y - trajectory.crossing_y)
        saved = error <= config.goalie_half_width
        saves += int(saved)
        absolute_errors.append(error)
        print(
            f"shot={shot_index + 1:02d} controller={args.controller} "
            f"actual={trajectory.crossing_y:+.3f} predicted={predicted_y:+.3f} "
            f"goalie={goalie_y:+.3f} error={error:.3f} "
            f"result={'SAVE' if saved else 'GOAL'}"
        )

        for _ in range(15):
            scene.step()
            if camera is not None and _ % 2 == 0:
                camera.render()

    if camera is not None:
        camera.stop_recording(save_to_filename=str(args.record), fps=25)
        print(f"Video: {args.record}")

    print(
        f"Summary: {saves}/{args.shots} saves ({saves / args.shots:.1%}), "
        f"mean_abs_error={np.mean(absolute_errors):.3f} m"
    )
    print(
        "Training validation: "
        f"AI MAE={training_metrics['ai_mae']:.3f} m, "
        f"linear MAE={training_metrics['baseline_mae']:.3f} m"
    )


if __name__ == "__main__":
    main()
