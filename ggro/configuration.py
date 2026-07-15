"""Typed model and benchmark presets for GGRO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPreset:
    base_model: str
    reward_model: str


@dataclass(frozen=True)
class TaskPreset:
    max_new_tokens: int
    entropy_threshold: float
    refinements_8b: int
    refinements_3b: int
    default_num_prompts: int
    prefill_attack: bool = False


MODEL_PRESETS: Dict[str, ModelPreset] = {
    "llama-3.1-8b": ModelPreset(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        reward_model="Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    ),
    "llama-3.2-3b": ModelPreset(
        base_model="meta-llama/Llama-3.2-3B-Instruct",
        reward_model="Ray2333/GRM-Llama3.2-3B-rewardmodel-ft",
    ),
}


TASK_PRESETS: Dict[str, TaskPreset] = {
    "hex-phi": TaskPreset(256, 1.5, 8, 8, 100, prefill_attack=True),
    "xstest": TaskPreset(256, 1.5, 8, 8, 250),
    "hh-rlhf": TaskPreset(128, 1.5, 8, 3, 150),
    "arc-challenge": TaskPreset(1024, 2.0, 8, 3, 100),
    "mmlu-pro": TaskPreset(1024, 2.0, 8, 8, 210),
}


@dataclass
class GGROConfig:
    """Runtime configuration for one GGRO benchmark run."""

    model: str
    task: str
    output_dir: Path = Path("outputs")
    cache_dir: Optional[Path] = None
    hf_token: Optional[str] = None
    seed: int = 1
    start_index: int = 0
    num_prompts: Optional[int] = None
    max_new_tokens: Optional[int] = None
    num_refinement_steps: Optional[int] = None
    entropy_threshold: Optional[float] = None
    nudge_top_k: int = 25
    nudge_temperature: float = 0.1
    nudge_weight: float = 1.0
    nudge_selection: str = "greedy"
    use_scale_weights: bool = True
    prefill_attack: Optional[bool] = None
    debug: bool = False

    def __post_init__(self) -> None:
        if self.model not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset: {self.model}")
        if self.task not in TASK_PRESETS:
            raise ValueError(f"Unknown task preset: {self.task}")
        if self.nudge_selection not in {"sample", "greedy"}:
            raise ValueError("nudge_selection must be 'sample' or 'greedy'")

        task = TASK_PRESETS[self.task]
        if self.num_prompts is None:
            self.num_prompts = task.default_num_prompts
        if self.max_new_tokens is None:
            self.max_new_tokens = task.max_new_tokens
        if self.entropy_threshold is None:
            self.entropy_threshold = task.entropy_threshold
        if self.prefill_attack is None:
            self.prefill_attack = task.prefill_attack
        if self.num_refinement_steps is None:
            self.num_refinement_steps = task.refinements_8b if self.model == "llama-3.1-8b" else task.refinements_3b

        self.output_dir = Path(self.output_dir)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)

    @property
    def model_preset(self) -> ModelPreset:
        return MODEL_PRESETS[self.model]

    @property
    def task_preset(self) -> TaskPreset:
        return TASK_PRESETS[self.task]
