"""SGLang modeling shared by baseline and LoopSpec execution."""

from __future__ import annotations

from collections.abc import Iterable
import logging
from typing import Any

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from .contracts import (
    RecurrentState,
    build_recurrent_spec,
    recurrence_count,
)
from .ouro import build_ouro, map_ouro_weight
from .raven import build_raven, map_raven_weight


logger = logging.getLogger(__name__)


def _collect_attention_layers(executor: nn.Module) -> tuple[RadixAttention, ...]:
    """Return block.self_attn.attn objects in the executor's block order.

    Executors without blocks and blocks without self_attn contribute nothing.
    These are references to the existing layers, not copies of the modules.
    """
    return tuple(
        block.self_attn.attn
        for block in getattr(executor, "blocks", ())
        if hasattr(block, "self_attn")
    )


class RecurrentServingModel(nn.Module):
    """Expose one executor per forward call for the LoopSpec runtime.

    Also owns model construction and weight loading shared with the baseline.
    LoopSpec supplies the role and manages recurrence and KV timelines externally.
    """

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Build family-specific executors from an HF model config.

        quant_config and prefix are passed to the builders for SGLang layers.
        Executor attribute names are also the targets of checkpoint mapping.
        """
        super().__init__()
        self.config = config
        self.spec = build_recurrent_spec(config)

        if self.spec.family == "raven":
            executors = build_raven(config, self.spec.block_layout, quant_config, prefix)
        elif self.spec.family == "ouro":
            executors = build_ouro(config, quant_config, prefix)
        else:
            raise ValueError(f"unsupported recurrent model family: {self.spec.family}")
        (self.pre_executor, self.recurrent_executor, self.post_executor) = executors

        # Keep references so the runtime can inspect or renumber native attention.
        self.attention_layers: dict[str, tuple[RadixAttention, ...]] = {
            "pre": _collect_attention_layers(self.pre_executor),
            "recurrent": _collect_attention_layers(self.recurrent_executor),
            "post": _collect_attention_layers(self.post_executor),
        }
        self._cuda_graph_inputs: dict[
            int, tuple[torch.Tensor, torch.Tensor | None]
        ] = {}
        # Exclusive layer range for one pre/core/post traversal. The baseline
        # subclass expands this range to cover every recurrent iteration.
        self.start_layer = 0
        self.end_layer = sum(map(len, self.attention_layers.values()))

    def cuda_graph_inputs(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return reusable (hidden, injected) input buffers for CUDA graphs.

        Tensors have shape [batch_size, hidden_size] and use the first model
        parameter's device/dtype. injected is None for Ouro. Repeated calls
        with the same size return the same buffers without clearing them.
        These are graph input buffers, not Raven's random hidden initializer.
        """
        buffers = self._cuda_graph_inputs.get(batch_size)
        if buffers is None:
            parameter = next(self.parameters())
            hidden = torch.zeros(
                (batch_size, self.config.hidden_size),
                dtype=parameter.dtype,
                device=parameter.device,
            )
            injected = (torch.zeros_like(hidden) if self.spec.family == "raven" else None)
            buffers = hidden, injected
            self._cuda_graph_inputs[batch_size] = buffers
        return buffers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> RecurrentState | LogitsProcessorOutput:
        """Dispatch the role attached to forward_batch by the LoopSpec runtime.

        input_ids/positions are [num_tokens]; optional input_embeds are
        [num_tokens, hidden_size] and are used only by pre. Pre and recurrent
        return RecurrentState; post returns LogitsProcessorOutput. Non-pre roles
        reconstruct state from the batch's hidden_states and injected fields.
        Additional SGLang keyword arguments are ignored.
        """
        del kwargs
        role = forward_batch.recurrent_role
        if role == "pre":
            return self.pre_executor(input_ids, positions, forward_batch, input_embeds=input_embeds)
        # Both eager and CUDA graph runners carry state tensors in the batch.
        recurrent_state = RecurrentState(
            hidden=forward_batch.hidden_states,
            positions=positions,
            injected=forward_batch.injected,
        )
        if role == "post":
            return self.post_executor(recurrent_state, input_ids, forward_batch)
        if role != "recurrent":
            raise ValueError(f"invalid recurrent serving role: {role}")
        return self.recurrent_executor(recurrent_state, forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load checkpoint (name, tensor) pairs into the model parameters.

        Map names before loading, including packed shards and shared weights.
        Report checkpoint tensors that cannot be matched; this does not check
        whether every model parameter was present in the checkpoint iterable.
        Parameter-specific loader errors propagate to the caller.
        """
        # Keep both embedding and LM-head names when they share a parameter.
        # This preserves aliases, not separate copies of the weight tensor.
        params = dict(self.named_parameters(remove_duplicate=False))
        unmatched: list[str] = []

        for name, weight in weights:
            if self.spec.family == "raven":
                target, shard_id = map_raven_weight(name, self.spec.block_layout)
            elif self.spec.family == "ouro":
                target, shard_id = map_ouro_weight(name)
            else:
                raise ValueError(f"unsupported recurrent model family: {self.spec.family}")

            parameter = params.get(target) if target is not None else None
            if parameter is None:
                unmatched.append(name)
                continue

            # SGLang loaders handle tensor-parallel slicing and packed shards.
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(parameter, weight)
            else:
                loader(parameter, weight, shard_id)

        if unmatched:
            message = (
                f"{len(unmatched)} recurrent checkpoint tensors were not matched; "
                f"first entries: {unmatched[:8]}"
            )
            if getattr(self.config, "strict_recurrent_weights", False):
                raise ValueError(message)
            logger.warning(message)


class LoopedRecurrentServingModel(RecurrentServingModel):
    """Run pre, all recurrent iterations, and post in one baseline forward.

    Recurrent iterations reuse weights but address distinct logical KV layers.
    """

    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        """Build shared modules and expose the unfolded logical layer count."""
        super().__init__(config, quant_config, prefix)
        self.recurrent_steps = recurrence_count(config)
        pre_layers = len(self.pre_executor.blocks)
        core_layers = len(self.recurrent_executor.blocks)
        post_layers = len(self.post_executor.blocks)
        self.end_layer = (pre_layers + self.recurrent_steps * core_layers + post_layers)

    def _number_attention(self, role: str, first_layer: int) -> int:
        """Assign consecutive layer IDs in place and return the next unused ID.

        SGLang uses these IDs to select each layer's KV cache. Renumbering the
        shared recurrent modules separates the cache of each logical iteration.
        """
        layers = self.attention_layers[role]
        for offset, layer in enumerate(layers):
            layer.layer_id = first_layer + offset
        return first_layer + len(layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """Run the complete fixed-depth model and return SGLang logits output.

        input_ids/positions are [num_tokens]; optional input_embeds are
        [num_tokens, hidden_size]. Unlike the split forward, this does not read
        recurrent_role: it runs pre once, recurrent_steps core iterations,
        and post once. Additional SGLang keyword arguments are ignored.
        """
        del kwargs
        next_layer = self._number_attention("pre", 0)
        state = self.pre_executor(input_ids, positions, forward_batch, input_embeds=input_embeds)
        for _ in range(self.recurrent_steps):
            # The weights stay shared; only the logical attention/KV IDs advance.
            next_layer = self._number_attention("recurrent", next_layer)
            state = self.recurrent_executor(state, forward_batch)
        self._number_attention("post", next_layer)
        return self.post_executor(state, input_ids, forward_batch)


# SGLang's model registry discovers these serving entry points.
EntryClass: list[type[RecurrentServingModel]] = [
    RecurrentServingModel,
    LoopedRecurrentServingModel,
]
