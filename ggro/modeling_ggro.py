"""Llama model wrapper that exposes GGRO's differentiable reward objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LogitsProcessorList,
    MaxLengthCriteria,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
)

from .generation import GGROGenerationMixin


@dataclass
class SegmentObjective:
    loss: torch.Tensor
    token_ids: torch.Tensor
    differentiable_tokens: torch.Tensor
    finished: bool
    base_logits: torch.Tensor
    per_example_loss: torch.Tensor


class GGRORewardObjectiveMixin:
    """Reward-model integration used to obtain token-level reward gradients."""

    reward_model: nn.Module

    def attach_reward_model(self, reward_model: nn.Module) -> None:
        self.reward_model = reward_model.eval()
        for parameter in self.reward_model.parameters():
            parameter.requires_grad_(False)

    def configure_ggro_generation(self, total_sequence_length: int) -> None:
        self.ggro_total_sequence_length = total_sequence_length
        eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        self.ggro_logits_processor = LogitsProcessorList(
            [MinLengthLogitsProcessor(total_sequence_length, eos_token_id=eos_token_id)]
        )
        self.ggro_stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=total_sequence_length)]
        )

    @staticmethod
    def prepend_prompt_biases(
        input_ids: torch.Tensor,
        nudge_biases: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Convert generated-only biases to the full-sequence layout used by decoding."""

        if nudge_biases is None:
            return None
        prompt_biases = torch.zeros(
            nudge_biases.shape[0],
            input_ids.shape[1],
            nudge_biases.shape[2],
            dtype=nudge_biases.dtype,
            device=nudge_biases.device,
        )
        return torch.cat([prompt_biases, nudge_biases], dim=1)

    def generate_and_score_segment(
        self,
        *,
        input_ids: torch.Tensor,
        llm_tokenizer,
        reward_tokenizer,
        nudge_biases: Optional[torch.Tensor],
        nudge_confidences: Optional[List[float]],
        use_scale_weights: bool,
        is_initial_segment: bool,
        entropy_threshold: float,
        total_sequence_length: int,
        debug: bool = False,
    ) -> SegmentObjective:
        """Generate one segment and differentiate its negative reward."""

        full_sequence_biases = self.prepend_prompt_biases(input_ids, nudge_biases)

        (
            output_ids,
            straight_through_tokens,
            _,
            soft_tokens,
            _,
            base_logits,
        ) = self.generate_ggro_segment(
            input_ids=input_ids.to(self.device),
            logits_processor=self.ggro_logits_processor,
            stopping_criteria=self.ggro_stopping_criteria,
            nudge_biases=full_sequence_biases.to(self.device) if full_sequence_biases is not None else None,
            sequence_length=self.ggro_total_sequence_length,
            tokenizer=reward_tokenizer,
            nudge_confidences=nudge_confidences,
            use_scale_weights=use_scale_weights,
            is_initial_segment=is_initial_segment,
            stop_on_high_entropy=True,
            entropy_threshold=entropy_threshold,
            total_sequence_length=total_sequence_length,
        )

        eos_token_id = llm_tokenizer.eos_token_id
        finished = bool(torch.all(output_ids[:, -1] == eos_token_id).item())
        if not finished and output_ids.shape[1] != total_sequence_length:
            output_ids = output_ids[:, :-1]
            straight_through_tokens = straight_through_tokens[:, :-1, :]
            soft_tokens = soft_tokens[:, :-1, :]
            base_logits = base_logits[:, :-1, :]

        differentiable_tokens = straight_through_tokens.detach().requires_grad_(True)
        soft_tokens = soft_tokens.detach().requires_grad_(True)
        reward_embedding = self.reward_model.get_input_embeddings().weight

        hard_embeddings = differentiable_tokens.to(torch.bfloat16).to(self.reward_model.device) @ reward_embedding
        hard_reward = self.reward_model(inputs_embeds=hard_embeddings).logits.squeeze(-1)

        # Preserve the original soft pass even though GGRO optimizes the hard objective.
        soft_embeddings = soft_tokens.to(torch.bfloat16).to(self.reward_model.device) @ reward_embedding
        _ = self.reward_model(inputs_embeds=soft_embeddings).logits.squeeze(-1)
        torch.cuda.empty_cache()

        if debug:
            print("Segment:", reward_tokenizer.decode(output_ids[0], skip_special_tokens=False))
            print("Reward:", hard_reward.detach().cpu().tolist())

        per_example_loss = -hard_reward
        return SegmentObjective(
            loss=per_example_loss.sum(),
            token_ids=output_ids,
            differentiable_tokens=differentiable_tokens,
            finished=finished,
            base_logits=base_logits,
            per_example_loss=per_example_loss,
        )


class GGROLlamaForCausalLM(
    GGRORewardObjectiveMixin,
    GGROGenerationMixin,
    LlamaForCausalLM,
):
    """Llama causal LM with project-local GGRO generation methods."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        kwargs.setdefault("torch_dtype", torch.bfloat16)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
