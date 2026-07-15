"""Summarize locally computable metrics from a GGRO JSONL output file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    summary = {
        "examples": len(rows),
        "mean_reward": float(np.mean([row["reward"] for row in rows])) if rows else None,
        "mean_nudges": float(np.mean([row["num_nudges"] for row in rows])) if rows else None,
    }
    if rows and all("correct" in row for row in rows):
        summary["accuracy"] = float(np.mean([row["correct"] for row in rows]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.path), indent=2))


if __name__ == "__main__":
    main()
