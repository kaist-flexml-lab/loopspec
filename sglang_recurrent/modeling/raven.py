"""Native SGLang modeling and weight mapping for Raven checkpoints."""

from copy import copy
from math import isfinite
from numbers import Real

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.llama import LlamaDecoderLayer
from sglang.srt.models.olmo2 import Olmo2DecoderLayer
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix

from .components import PostExecutor, PreExecutor


_RAVEN_ROOT_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "transformer.prelude.": "pre_executor.blocks.",
        "transformer.core_block.": "recurrent_executor.blocks.",
        "transformer.coda.": "post_executor.blocks.",
        "transformer.wte.weight": "pre_executor.embed_tokens.weight",
        "transformer.adapter.weight": "recurrent_executor.adapter.weight",
        "transformer.adapter.bias": "recurrent_executor.adapter.bias",
        "transformer.ln_f.weight": "post_executor.final_norm.weight",
        "lm_head.weight": "post_executor.lm_head.weight",
    }
)
_RAVEN_COMPONENT_MAPPERS = {
    layout: WeightsMapper(
        orig_to_new_substr={
            ".attn.Wqkv.": ".self_attn.qkv_proj.",
            ".attn.proj.": ".self_attn.o_proj.",
            ".attn.q_norm.": ".self_attn.q_norm.",
            ".attn.k_norm.": ".self_attn.k_norm.",
            ".mlp.fc.": ".mlp.gate_up_proj.",
            ".mlp.proj.": ".mlp.down_proj.",
            **norm_replacements,
        }
    )
    for layout, norm_replacements in {
        "raven_pre": {
            ".norm_1.": ".input_layernorm.",
            ".norm_2.": ".post_attention_layernorm.",
        },
        "olmo_post": {
            ".norm_1.": ".post_attention_layernorm.",
            ".norm_2.": ".post_feedforward_layernorm.",
        },
    }.items()
}


class MaterializedLlamaDecoderLayer(LlamaDecoderLayer):
    """Native Llama layer with its deferred residual materialized."""

    def forward(self, positions, hidden_states, forward_batch):
        hidden_states, residual = super().forward(
            positions, hidden_states, forward_batch, residual=None
        )
        return hidden_states + residual


class RavenRecurrentExecutor(nn.Module):
    def __init__(self, adapter, blocks):
        super().__init__()
        self.adapter = adapter
        self.blocks = blocks

    def forward(self, state, forward_batch):
        hidden, _ = self.adapter(
            torch.cat((state.hidden, state.injected), dim=-1)
        )
        for block in self.blocks:
            hidden = block(state.positions, hidden, forward_batch)
        state.hidden = hidden
        return state


def _validate_raven_serving_config(config):
    """Reject test-time noise until native serving implements it."""
    noise = getattr(config, "test_time_noise", 0)
    if (
        isinstance(noise, bool)
        or not isinstance(noise, Real)
        or not isfinite(float(noise))
        or float(noise) != 0.0
    ):
        raise ValueError(
            "Raven serving does not support test_time_noise; "
            f"expected a finite numeric zero, got {noise!r}"
        )


def _raven_decoder_config(config, block_layout):
    """Map Raven's remote config names to native SGLang decoder fields."""
    native = copy(config)
    native.rms_norm_eps = config.norm_eps
    native.hidden_act = "silu"
    native.attention_bias = bool(config.bias)
    if block_layout == "olmo_post":
        rope_parameters = dict(
            getattr(config, "rope_parameters", None) or {}
        )
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(config, "rope_base", None)
        if rope_theta is None:
            raise ValueError("OLMo-layout Raven config requires a RoPE base")
        rope_parameters.setdefault("rope_theta", float(rope_theta))
        rope_parameters.setdefault("rope_type", "default")
        native.rope_parameters = rope_parameters
        native.rope_scaling = dict(
            getattr(config, "rope_scaling", None) or rope_parameters
        )
    return native


def build_raven(config, block_layout, quant_config, prefix):
    _validate_raven_serving_config(config)
    native = _raven_decoder_config(config, block_layout)
    block_class = (
        MaterializedLlamaDecoderLayer
        if block_layout == "raven_pre"
        else Olmo2DecoderLayer
    )

    def blocks(count, name, layer_offset=0):
        return nn.ModuleList(
            block_class(
                config=native,
                layer_id=layer_offset + index,
                quant_config=quant_config,
                prefix=add_prefix(f"{name}.{index}", prefix),
            )
            for index in range(count)
        )

    embed_tokens = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("embed_tokens", prefix),
    )
    pre = PreExecutor(
        embed_tokens,
        blocks(config.n_layers_in_prelude, "pre"),
        embed_scale=float(config.init_values["embed_scale"]),
        init_std=float(config.init_values["std"]),
    )
    recurrent = RavenRecurrentExecutor(
        ColumnParallelLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=bool(config.bias),
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("adapter", prefix),
        ),
        blocks(
            config.n_layers_in_recurrent_block,
            "recurrent",
            config.n_layers_in_prelude,
        ),
    )
    post_blocks = blocks(
        config.n_layers_in_coda,
        "post",
        config.n_layers_in_prelude + config.n_layers_in_recurrent_block,
    )
    final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
    lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("lm_head", prefix),
    )
    if config.tie_embeddings:
        lm_head.tie_weights(embed_tokens)
    post = PostExecutor(post_blocks, final_norm, lm_head, config)
    return pre, recurrent, post


def map_raven_weight(name, block_layout):
    target = _RAVEN_ROOT_MAPPER.apply_list([name])[0]
    if target == name:
        return None, None
    return _RAVEN_COMPONENT_MAPPERS[block_layout].apply_list([target])[0], None
