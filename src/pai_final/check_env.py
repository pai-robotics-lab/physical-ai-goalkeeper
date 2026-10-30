"""Validate the local runtime before starting simulation work."""

from __future__ import annotations

import argparse
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the Physical AI runtime")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    import torch

    print(f"Python:  {sys.version.split()[0]}")
    print(f"OS:      {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")

    if args.backend == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA backend was requested, but PyTorch cannot use CUDA.")
        return 1

    import genesis as gs

    version = getattr(gs, "__version__", "unknown")
    print(f"Genesis: {version}")
    print("Environment check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
