"""Native SGLang modeling and weight mapping for Ouro checkpoints."""

from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.models.gemma2 import Gemma2DecoderLayer
from sglang.srt.models.llama import LlamaAttention, LlamaMLP
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.utils import add_prefix

from .components import PostExecutor, PreExecutor, map_packed_weight


_OURO_ROOT_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.embed_tokens.weight": "pre_executor.embed_tokens.weight",
        "model.norm.weight": "recurrent_executor.step_norm.weight",
        "model.early_exit_gate.": "recurrent_executor.early_exit_gate.",
        "model.layers.": "recurrent_executor.blocks.",
        "lm_head.weight": "post_executor.lm_head.weight",
    }
)
_OURO_NORM_MAPPER = WeightsMapper(
    orig_to_new_substr={
        ".input_layernorm_2.": ".post_attention_layernorm.",
        ".post_attention_layernorm.": ".pre_feedforward_layernorm.",
        ".post_attention_layernorm_2.": ".post_feedforward_layernorm.",
    }
)


class OuroDecoderLayer(Gemma2DecoderLayer):
    """Ouro components with SGLang's complete Gemma2 residual flow."""

    def __init__(self, config, layer_id, quant_config, prefix):
        # Ouro shares Gemma2's residual flow, not its attention, GELU MLP, or
        # normalization parameters, so initialize only the common Module base.
        nn.Module.__init__(self)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
            rope_is_neox_style=True,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=False,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions, hidden_states, forward_batch):
        hidden_states, residual = Gemma2DecoderLayer.forward(
            self, positions, hidden_states, forward_batch, residual=None
        )
        return hidden_states + residual


class OuroRecurrentExecutor(nn.Module):
    def __init__(self, blocks, step_norm, early_exit_gate):
        super().__init__()
        self.blocks = blocks
        self.step_norm = step_norm
        # Fixed-depth serving retains but never evaluates this checkpoint head.
        self.early_exit_gate = early_exit_gate

    def forward(self, state, forward_batch):
        hidden = state.hidden
        for block in self.blocks:
            hidden = block(state.positions, hidden, forward_batch)
        state.hidden = self.step_norm(hidden)
        return state


def build_ouro(config, quant_config, prefix):
    embed_tokens = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("embed_tokens", prefix),
    )
    recurrent = OuroRecurrentExecutor(
        nn.ModuleList(
            OuroDecoderLayer(
                config,
                index,
                quant_config,
                add_prefix(f"recurrent.{index}", prefix),
            )
            for index in range(config.num_hidden_layers)
        ),
        RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        ReplicatedLinear(
            config.hidden_size,
            1,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("early_exit_gate", prefix),
        ),
    )
    lm_head = ParallelLMHead(
        config.vocab_size,
        config.hidden_size,
        quant_config=quant_config,
        prefix=add_prefix("lm_head", prefix),
    )
    if config.tie_word_embeddings:
        lm_head.tie_weights(embed_tokens)
    post = PostExecutor(nn.ModuleList(), None, lm_head, config)
    return PreExecutor(embed_tokens, nn.ModuleList()), recurrent, post


def map_ouro_weight(name):
    target = _OURO_ROOT_MAPPER.apply_list([name])[0]
    if target == name:
        return None, None
    target = _OURO_NORM_MAPPER.apply_list([target])[0]
    return map_packed_weight(target)
