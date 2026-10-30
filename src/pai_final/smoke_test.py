"""Run a small headless Genesis simulation as an installation smoke test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Genesis installation smoke test")
    parser.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), default="cpu")
    args = parser.parse_args()

    # Keep both current Quadrants and legacy GsTaichi caches inside the repository.
    cache_name = "quadrants" if args.backend == "cpu" else f"quadrants-{args.backend}"
    os.environ.setdefault(
        "QD_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / cache_name),
    )
    os.environ.setdefault(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / f"gstaichi-{args.backend}"),
    )

    import genesis as gs

    backend = getattr(gs, args.backend, None)
    if backend is None:
        raise RuntimeError(
            f"Genesis {getattr(gs, '__version__', 'unknown')} does not support "
            f"the {args.backend!r} backend."
        )
    gs.init(backend=backend, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(pos=(0.0, 0.0, 1.0), radius=0.08),
    )
    scene.build()

    initial_z = float(sphere.get_pos()[-1])
    for _ in range(30):
        scene.step()
    final_z = float(sphere.get_pos()[-1])

    if not final_z < initial_z:
        raise RuntimeError(f"Sphere did not fall: initial_z={initial_z}, final_z={final_z}")

    print(
        f"Genesis {args.backend} smoke test: PASS "
        f"(z: {initial_z:.3f} -> {final_z:.3f})"
    )


if __name__ == "__main__":
    main()
