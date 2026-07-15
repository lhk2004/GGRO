"""Gradient-Guided Reward Optimization for inference-time alignment."""

from .configuration import GGROConfig, MODEL_PRESETS, TASK_PRESETS
from .modeling_ggro import GGROLlamaForCausalLM
from .nudging import GradientNudgeSelector

__all__ = [
    "GGROConfig",
    "GGROLlamaForCausalLM",
    "GradientNudgeSelector",
    "MODEL_PRESETS",
    "TASK_PRESETS",
]

__version__ = "1.0.0"
