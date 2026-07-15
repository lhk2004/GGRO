"""Project-local generation methods used by GGRO.

The setup portion of :meth:`generate_ggro_segment` is adapted from the
Apache-2.0 licensed Hugging Face Transformers 4.52.1 ``GenerationMixin``.
Keeping it here avoids modifying an installed Transformers package.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from transformers.cache_utils import Cache
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.utils import logging

logger = logging.get_logger(__name__)

SUPPORTED_TRANSFORMERS_VERSION = "4.52.1"


def require_supported_transformers() -> None:
    """Fail early when private generation APIs do not match the pinned release."""

    if transformers.__version__ != SUPPORTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "GGRO requires transformers==4.52.1 because its local generation "
            f"mixin uses private GenerationMixin APIs; found {transformers.__version__}."
        )


class GGROGenerationMixin:
    """Llama-only segmented decoding with optional GGRO nudging-token biases."""

    def _get_legacy_cache_position(self, input_ids, model_kwargs):
        """Prepare cache positions using the behavior of the original GGRO patch."""

        if (
            "inputs_embeds" in model_kwargs
            and model_kwargs["inputs_embeds"] is not None
            and not self.config.is_encoder_decoder
        ):
            cache_position = (
                torch.ones_like(model_kwargs["inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        elif "decoder_inputs_embeds" in model_kwargs and self.config.is_encoder_decoder:
            cache_position = (
                torch.ones_like(model_kwargs["decoder_inputs_embeds"][0, :, 0], dtype=torch.int64).cumsum(0) - 1
            )
        else:
            cache_position = torch.ones_like(input_ids[0, :], dtype=torch.int64).cumsum(0) - 1

        if model_kwargs.get("past_key_values") is not None:
            cache = model_kwargs["past_key_values"]
            past_length = 0
            if not isinstance(cache, Cache):
                past_length = cache[0][0].shape[2]
            elif hasattr(cache, "get_seq_length") and cache.get_seq_length() is not None:
                past_length = cache.get_seq_length()
            cache_position = cache_position[past_length:]

        model_kwargs["cache_position"] = cache_position
        return model_kwargs

    @torch.no_grad()
    def generate_ggro_segment(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.LongTensor, Tuple[torch.Tensor, ...]]:
        """Generate one entropy-bounded segment, optionally with a nudging token.

        This is the repository-local replacement for the original patched
        generation entry point.
        """

        require_supported_transformers()
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            raise ValueError("`tokenizer` is required for GGRO segment generation.")
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)

        ggro_kwargs = {
            "nudge_biases": kwargs.pop("nudge_biases", None),
            "use_scale_weights": kwargs.pop("use_scale_weights", False),
            "reverse": kwargs.pop("reverse", False),
            "sequence_length": kwargs.pop("sequence_length", None),
            "do_sample": kwargs.pop("do_sample", False),
            "nudge_confidences": kwargs.pop("nudge_confidences", None),
            "is_initial_segment": kwargs.pop("is_initial_segment", False),
            "stop_on_high_entropy": kwargs.pop("stop_on_high_entropy", False),
            "entropy_threshold": kwargs.pop("entropy_threshold", 1.0),
            "total_sequence_length": kwargs.pop("total_sequence_length", None),
            "tokenizer": tokenizer,
        }
        if "max_length" not in kwargs and ggro_kwargs["sequence_length"] is not None:
            kwargs["max_length"] = ggro_kwargs["sequence_length"]

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters)
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask") is not None

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]
        device = inputs_tensor.device
        generation_config.pad_token_id = tokenizer.pad_token_id
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        if (
            not self.config.is_encoder_decoder
            and generation_config._pad_token_tensor is not None
            and batch_size > 1
            and len(inputs_tensor.shape) == 2
            and inputs_tensor.shape[1] > 0
            and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
        ):
            logger.warning(
                "Right padding was detected for a decoder-only model. Set the tokenizer's padding_side to 'left'."
            )

        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask and model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
            raise ValueError("`attention_mask` passed to generation must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)
        if streamer is not None:
            streamer.put(input_ids.cpu())

        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        if streamer is not None and generation_config.num_beams > 1:
            raise ValueError("`streamer` cannot be used with beam search.")
        if self.device.type != input_ids.device.type:
            warnings.warn(
                f"Input IDs are on {input_ids.device.type}, but the model is on {self.device.type}.",
                UserWarning,
            )

        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        generation_config.stop_strings = "<|eot_id|>"
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
            **kwargs,
        )
        model_kwargs["use_cache"] = generation_config.use_cache

        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        return self._decode_ggro_segment(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **ggro_kwargs,
            **model_kwargs,
        )

    def _decode_ggro_segment(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer,
        nudge_biases: Optional[torch.Tensor] = None,
        use_scale_weights: bool = False,
        reverse: bool = False,
        sequence_length: Optional[int] = None,
        do_sample: bool = False,
        nudge_confidences: Optional[List[float]] = None,
        is_initial_segment: bool = False,
        stop_on_high_entropy: bool = False,
        entropy_threshold: float = 1.0,
        total_sequence_length: Optional[int] = None,
        tokenizer=None,
        **model_kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """Run the token loop used for GGRO segments."""

        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config._eos_token_tensor
        batch_size, current_length = input_ids.shape
        prompt_length = current_length
        peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_legacy_cache_position(input_ids, model_kwargs)

        if streamer is not None:
            raise ValueError("GGRO segment generation does not support streaming.")
        if nudge_biases is not None and total_sequence_length is None:
            raise ValueError("`total_sequence_length` is required when nudge biases are supplied.")

        vocab_size = self.config.vocab_size
        initial_one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size).float()
        soft_tokens = initial_one_hot
        logits_sequence = initial_one_hot
        base_logits_sequence = initial_one_hot
        straight_through_tokens = initial_one_hot
        next_token_scores = initial_one_hot[:, -1, :]
        last_token = None

        if sequence_length is None and nudge_biases is not None:
            sequence_length = nudge_biases.shape[1]

        while self._has_unfinished_sequences(peer_finished, synced_gpus, device=input_ids.device):
            if total_sequence_length is not None and input_ids.shape[1] >= total_sequence_length:
                break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, return_dict=True)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if synced_gpus and peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :]
            if nudge_biases is not None:
                # The GGRO path deliberately uses raw logits. The straight-through
                # expression preserves the original gradient behavior.
                raw_scores = LogitsProcessorList()(input_ids, next_token_logits.detach())
                next_token_scores = raw_scores.detach() + next_token_logits - next_token_logits.detach()
            else:
                for processor in logits_processor:
                    if isinstance(processor, TopKLogitsWarper):
                        processor.top_k = 40
                    elif isinstance(processor, TopPLogitsWarper):
                        processor.top_p = 1.0
                next_token_scores = logits_processor(input_ids, next_token_logits)

            base_logits = next_token_scores.clone()
            base_logits_sequence = torch.cat((base_logits_sequence, base_logits.unsqueeze(1)), dim=1)

            if nudge_biases is not None:
                bias_index = current_length if not reverse else sequence_length - current_length
                if use_scale_weights and nudge_biases.mean().item() != 0 and nudge_biases.shape[1] > bias_index:
                    logit_norms = next_token_logits.detach().norm(dim=-1, p=2)
                    bias_norms = nudge_biases[:, bias_index, :].detach().norm(dim=-1, p=2)
                    scaling_ratio = (logit_norms / bias_norms).unsqueeze(-1)
                else:
                    scaling_ratio = 1.0

                generated_offset = current_length - prompt_length
                insertion_offset = 4 if is_initial_segment else 0
                should_insert = generated_offset == insertion_offset
                if nudge_biases.shape[1] > bias_index and nudge_confidences is not None and should_insert:
                    # Preserve the forcing coefficient used by the released experiments.
                    next_token_scores = next_token_scores + 10_000_000.0 * scaling_ratio * nudge_biases[:, bias_index, :]

            if do_sample:
                probabilities = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            token_probabilities = torch.nn.functional.softmax(next_token_scores, dim=-1)
            token_one_hot = torch.nn.functional.one_hot(next_tokens, num_classes=vocab_size).float()
            straight_through = token_one_hot - token_probabilities.detach() + token_probabilities
            soft_tokens = torch.cat((soft_tokens, token_probabilities.unsqueeze(1)), dim=1)
            straight_through_tokens = torch.cat((straight_through_tokens, straight_through.unsqueeze(1)), dim=1)
            logits_sequence = torch.cat((logits_sequence, next_token_scores.unsqueeze(1)), dim=1)

            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            peer_finished = unfinished_sequences.max() == 0
            current_length += 1
            del outputs

            if stop_on_high_entropy and current_length - prompt_length >= 4:
                probabilities = torch.softmax(base_logits_sequence[:, -1, :], dim=-1).clamp_min(1e-9)
                entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean().item()
                if entropy >= entropy_threshold:
                    break

            if tokenizer is not None:
                last_token = tokenizer.decode(next_tokens)
                if not isinstance(last_token, str):
                    raise TypeError("Tokenizer.decode must return a string.")

        return (
            input_ids,
            straight_through_tokens,
            next_token_scores,
            soft_tokens,
            logits_sequence,
            base_logits_sequence,
        )
