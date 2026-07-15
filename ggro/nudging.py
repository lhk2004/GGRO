"""Gradient-informed nudging-token selection for GGRO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from .modeling_ggro import SegmentObjective

EPSILON = 1e-10


@dataclass
class NudgeProposal:
    biases: torch.Tensor
    confidences: List[float]
    token_ids: torch.Tensor
    objective: SegmentObjective


class GradientNudgeSelector(nn.Module):
    """Select nudging tokens from reward gradients over likely base-LM tokens."""

    def __init__(
        self,
        *,
        nudge_weight: float,
        nudge_temperature: float,
        nudge_top_k: int,
        device: str,
        selection: str = "greedy",
    ) -> None:
        super().__init__()
        self.nudge_weight = float(nudge_weight)
        self.nudge_temperature = float(nudge_temperature)
        self.nudge_top_k = int(nudge_top_k)
        self.device = str(device)
        self.selection = selection
        self.prompt_length = 0
        self.embedding = None

    def initialize_nudge_biases(
        self,
        *,
        model,
        batch_size: int,
        prompt_length: int,
        total_sequence_length: int,
    ) -> torch.Tensor:
        self.prompt_length = prompt_length
        self.embedding = model.get_input_embeddings()
        model.configure_ggro_generation(total_sequence_length)
        max_new_tokens = total_sequence_length - prompt_length
        return torch.zeros(
            batch_size,
            max_new_tokens,
            self.embedding.weight.shape[0],
            device=self.device,
            requires_grad=True,
        )

    def compute_reward_gradients(self, objective: SegmentObjective) -> torch.Tensor:
        gradients = torch.autograd.grad(objective.loss, objective.differentiable_tokens)[0].detach()
        return gradients[:, self.prompt_length :, :]

    def compute_nudge_candidate_scores(
        self,
        objective: SegmentObjective,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gradients = self.compute_reward_gradients(objective)
        current_token_ids = objective.token_ids[:, self.prompt_length :]
        replacement_mask = torch.ones_like(gradients, device=self.device)
        replacement_mask[
            torch.arange(replacement_mask.shape[0])[:, None, None],
            torch.arange(replacement_mask.shape[1])[None, :, None],
            current_token_ids.unsqueeze(-1),
        ] = EPSILON
        gradient_scores = -(gradients * replacement_mask)

        base_logits = objective.base_logits[:, self.prompt_length :, :]
        top_k = min(self.nudge_top_k, base_logits.shape[-1])
        candidate_token_ids = torch.topk(base_logits, top_k, dim=-1).indices
        candidate_scores = gradient_scores.gather(-1, candidate_token_ids)
        return candidate_scores, candidate_token_ids

    def propose_nudge_tokens(self, objective: SegmentObjective) -> tuple[torch.Tensor, List[float]]:
        candidate_scores, candidate_token_ids = self.compute_nudge_candidate_scores(objective)
        scaled_scores = candidate_scores / self.nudge_temperature
        if self.selection == "sample":
            selected_indices = torch.distributions.Categorical(logits=scaled_scores).sample()
        else:
            selected_indices = scaled_scores.argmax(dim=-1)
        selected_tokens = candidate_token_ids.gather(-1, selected_indices.unsqueeze(-1)).squeeze(-1)

        probabilities = torch.softmax(scaled_scores, dim=-1)
        confidences = probabilities.max(dim=-1).values[0].detach().cpu().tolist()
        return selected_tokens, confidences

    def build_embedding_distance_bias(self, nudge_token_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("initialize_nudge_biases must be called first")
        with torch.no_grad():
            nudge_embeddings = self.embedding(nudge_token_ids)
            vocabulary_norms = torch.einsum("ve->v", self.embedding.weight**2)[None, None, :]
            cross_terms = torch.einsum("bse,ve->bsv", nudge_embeddings, self.embedding.weight)
            nudge_norms = torch.einsum("bse->bs", nudge_embeddings**2).unsqueeze(-1)
            return -self.nudge_weight * (vocabulary_norms - 2 * cross_terms + nudge_norms)

    def refine_segment(self, objective: SegmentObjective) -> NudgeProposal:
        token_ids, confidences = self.propose_nudge_tokens(objective)
        return NudgeProposal(
            biases=self.build_embedding_distance_bias(token_ids),
            confidences=confidences,
            token_ids=token_ids,
            objective=objective,
        )
