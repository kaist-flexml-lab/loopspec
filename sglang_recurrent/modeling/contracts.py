"""Lightweight state and construction contracts for recurrent models."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class RecurrentState:
    """Tensors transferred between the pre, recurrent, and post executors.

    Executors may replace the hidden field as recurrence advances.
    This object contains no KV cache; ForwardBatch and the runtime
    provide the attention metadata and cache storage separately.
    """

    # Current token representations, shaped [num_tokens, hidden_size].
    hidden: torch.Tensor
    # Position within the sequence for each token, shaped [num_tokens].
    positions: torch.Tensor
    # Raven's pre output, shaped like hidden, reused as recurrent input.
    # Ouro does not use a separate injection tensor and leaves this as None.
    injected: torch.Tensor | None = None


@dataclass(frozen=True)
class RecurrentModelSpec:
    """Construction-time description of a recurrent checkpoint."""

    family: str  # "raven" or "ouro"; selects the model builder.
    block_layout: str | None = None  # Raven: "raven_pre" / "olmo_post".


def build_recurrent_spec(config: Any) -> RecurrentModelSpec:
    """Return the model family and Raven decoder layout from an HF config.

    Reads model_type and, for Raven, head_dim. Ouro has no layout variant.
    Raises ValueError for an unsupported model_type.
    """
    if config.model_type == "ouro":
        return RecurrentModelSpec("ouro")
    if config.model_type != "huginn_raven":
        raise ValueError(
            f"unsupported recurrent model type: {config.model_type}"
        )
    # TODO: Replace this checkpoint-specific heuristic with explicit layout
    # metadata. The released Raven configs call both the Llama pre-norm and
    # OLMo2 post-norm implementations "SandwichBlock" and do not identify
    # which implementation they contain; head_dim only happens to distinguish
    # the currently supported checkpoints.
    # "olmo_post": OLMo2-based Raven; normalize attention/MLP outputs before
    # adding each residual: x = x + norm_1(attn(x)); x = x + norm_2(mlp(x)).
    # "raven_pre": Llama/TinyLlama-based Raven; normalize attention/MLP inputs:
    # x = x + attn(norm_1(x)); x = x + mlp(norm_2(x)).
    # Here pre/post describes norm placement within a block, not executor roles.
    layout = "olmo_post" if config.head_dim == 128 else "raven_pre"
    return RecurrentModelSpec("raven", layout)


def recurrence_count(config: Any) -> int:
    """Return the fixed number of recurrent iterations used during serving.

    Reads total_ut_steps for Ouro and mean_recurrence for Raven from an HF config,
    converting the value to int. This is an iteration count, not a layer count;
    it does not sample Raven's training-time recurrence distribution.
    """
    if config.model_type == "ouro":
        return int(config.total_ut_steps)
    return int(config.mean_recurrence)


__all__ = [
    "RecurrentModelSpec",
    "RecurrentState",
    "build_recurrent_spec",
    "recurrence_count",
]
