"""Command-line interface for GGRO experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .configuration import GGROConfig, MODEL_PRESETS, TASK_PRESETS
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ggro-run",
        description="Run Gradient-Guided Reward Optimization on a paper benchmark.",
    )
    parser.add_argument("--model", choices=MODEL_PRESETS, required=True)
    parser.add_argument("--task", choices=TASK_PRESETS, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--num-refinement-steps", type=int)
    parser.add_argument("--entropy-threshold", type=float)
    parser.add_argument("--nudge-top-k", type=int, default=25)
    parser.add_argument("--nudge-temperature", type=float, default=0.1)
    parser.add_argument("--nudge-weight", type=float, default=1.0)
    parser.add_argument("--nudge-selection", choices=("sample", "greedy"), default="greedy")
    parser.add_argument(
        "--scale-nudge-biases",
        dest="use_scale_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prefill-attack", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = run_benchmark(GGROConfig(**vars(args)))
    print(output_path)


if __name__ == "__main__":
    main()
