"""Shared model components and packed-weight mapping."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.utils import WeightsMapper

from .contracts import RecurrentState


# Each split checkpoint projection maps to a packed parameter and its shard.
# Q/K/V share one projection; gate/up share another. No tensors are packed here.
_PACKED_WEIGHT_RULES = {
    ".self_attn.q_proj.": (".self_attn.qkv_proj.", "q"),
    ".self_attn.k_proj.": (".self_attn.qkv_proj.", "k"),
    ".self_attn.v_proj.": (".self_attn.qkv_proj.", "v"),
    ".mlp.gate_proj.": (".mlp.gate_up_proj.", 0),
    ".mlp.up_proj.": (".mlp.gate_up_proj.", 1),
}
# The mapper only renames parameters; the weight loader consumes the shard ID.
_PACKED_WEIGHT_MAPPER = WeightsMapper(
    orig_to_new_substr={
        source: destination
        for source, (destination, _shard) in _PACKED_WEIGHT_RULES.items()
    }
)


def map_packed_weight(name: str) -> tuple[str, str | int | None]:
    """Return the packed parameter name and shard for a checkpoint name.

    Q/K/V shards are "q"/"k"/"v"; gate/up shards are 0/1. Names without a
    split projection are returned unchanged with shard None.
    """
    for source, (_destination, shard) in _PACKED_WEIGHT_RULES.items():
        if source in name:
            return _PACKED_WEIGHT_MAPPER.apply_list([name])[0], shard
    return name, None


class PreExecutor(nn.Module):
    """Embed, apply optional pre blocks, and create the recurrent state."""

    def __init__(
        self,
        embed_tokens: nn.Module,
        blocks: nn.ModuleList,
        *,
        embed_scale: float = 1.0,
        init_std: float | None = None,
    ) -> None:
        """Store the embedding, pre blocks, and state-init settings.

        Pass an empty ModuleList when there are no pre blocks.
        ``embed_tokens`` maps token IDs to hidden vectors and must expose
        ``embedding_dim`` when separate hidden-state initialization is used.
        ``init_std=None`` uses the pre output as recurrent hidden (Ouro).
        A numeric init_std creates a separate hidden state (Raven), while the
        scaled pre output is retained as the recurrent injection.
        """
        super().__init__()
        self.embed_tokens = embed_tokens
        self.blocks = blocks
        self.embed_scale = embed_scale
        self.init_std = init_std

    @property
    def has_separate_hidden_state(self) -> bool:
        """Return whether recurrent hidden is initialized apart from the input."""
        return self.init_std is not None

    def make_initial_hidden(
        self,
        count: int,
        *,
        device: torch.device | str | int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return Raven hidden shaped [count, embedding_dim] on device/dtype.

        The released checkpoint draws ``randn`` and then overwrites it with
        ``trunc_normal_``. Both draws consume the device's default RNG stream
        to preserve the checkpoint's random-number consumption order.

        Positive init_std uses a normal distribution truncated at +/-3 std;
        nonpositive init_std produces zeros. Apply embed_scale afterward.
        Raises RuntimeError when separate hidden initialization is disabled.
        """
        if self.init_std is None:
            raise RuntimeError("this pre executor has no separate hidden state")
        hidden = torch.randn(
            (count, self.embed_tokens.embedding_dim),
            device=device,
            dtype=dtype,
        )
        if self.init_std > 0:
            nn.init.trunc_normal_(
                hidden,
                mean=0.0,
                std=self.init_std,
                a=-3 * self.init_std,
                b=3 * self.init_std,
            )
        else:
            hidden.zero_()
        if self.embed_scale != 1:
            hidden.mul_(self.embed_scale)
        return hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> RecurrentState:
        """Build recurrent state from token IDs or supplied token embeddings.

        IDs and positions have shape [num_tokens]; input_embeds, when supplied,
        has shape [num_tokens, hidden_size] and bypasses embedding lookup.
        forward_batch carries SGLang's attention and KV-cache metadata.
        Returns hidden and optional injected tensors with the original positions.
        """
        hidden = (self.embed_tokens(input_ids) if input_embeds is None else input_embeds)
        if self.embed_scale != 1:
            hidden = hidden * self.embed_scale
        for block in self.blocks:
            hidden = block(positions, hidden, forward_batch)
        if not self.has_separate_hidden_state:
            # Ouro starts recurrence directly from the embedded input.
            return RecurrentState(hidden=hidden, positions=positions)
        # Raven keeps the pre output as input to every recurrent iteration.
        recurrent_hidden = self.make_initial_hidden(
            hidden.shape[0],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return RecurrentState(
            hidden=recurrent_hidden,
            injected=hidden,
            positions=positions,
        )


class PostExecutor(nn.Module):
    """Apply post blocks and normalization, then produce SGLang logits."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        final_norm: nn.Module | None,
        lm_head: ParallelLMHead,
        config: Any,
    ) -> None:
        """Store post layers and an externally constructed vocabulary head.

        Pass an empty ModuleList when there are no post blocks.
        blocks run before the optional final_norm. The builder configures
        lm_head, including any embedding weight sharing. config supplies
        vocabulary size and optional softcapping to the logits processor.
        """
        super().__init__()
        self.blocks = blocks
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        state: RecurrentState,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Convert recurrent hidden into SGLang's logits output object.

        state carries hidden [num_tokens, hidden_size] and token positions;
        input_ids has shape [num_tokens]. forward_batch selects the output
        positions and optional log-probability/hidden-state fields.
        Returns LogitsProcessorOutput, not a bare logits tensor.
        """
        hidden_states = state.hidden
        for block in self.blocks:
            hidden_states = block(state.positions, hidden_states, forward_batch)
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        # SGLang selects the needed token rows and applies the vocabulary head.
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)


__all__ = ["PostExecutor", "PreExecutor", "map_packed_weight"]
