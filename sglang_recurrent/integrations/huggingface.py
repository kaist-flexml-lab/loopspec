"""Transformers configurations for recurrent checkpoint compatibility."""

from math import sqrt
from typing import Any

from transformers import AutoConfig, PreTrainedConfig


def align_raven_special_token_ids(
    config: Any,
    tokenizer: Any,
    *mirrors: Any,
) -> Any:
    """Align one Raven config with its checkpoint tokenizer in place.

    Released Raven model and generation configs retain stale Huginn token IDs
    after their weights are paired with Llama, OLMo, or TinyLlama tokenizers.
    Callers own the config instance they pass here: SGLang already deep-copies
    its model config, while planning integrations should pass a shallow copy.
    """

    if getattr(config, "model_type", None) != RavenConfig.model_type:
        return config
    if tokenizer is None:
        raise ValueError("Raven special-token alignment requires the target tokenizer")
    vocab_size = getattr(config, "padded_vocab_size", None) or getattr(config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("Raven target config must define a positive vocabulary")

    resolved = {}
    for name in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        token_id = getattr(tokenizer, name, None)
        if token_id is not None and (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocab_size
        ):
            raise ValueError(
                f"Raven tokenizer {name}={token_id!r} is outside target "
                f"vocabulary size {vocab_size}"
            )
        resolved[name] = token_id
    for target in (config, *mirrors):
        for name, token_id in resolved.items():
            setattr(target, name, token_id)

    return config


class RavenConfig(PreTrainedConfig):
    """Load released Raven checkpoints through the Transformers 5 API.

    Released checkpoints ship a Transformers 4-era remote config. Transformers
    5 standardizes RoPE settings before assigning arbitrary keyword arguments,
    so the remote class fails while reading Llama-3 scaling parameters. Keeping
    the compatibility class local also avoids patching downloaded model files.
    """

    model_type = "huginn_raven"
    keys_to_ignore_at_inference = [""]
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "n_embd",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        n_embd: int = 5280,
        n_heads: int = 55,
        n_layers: int = 8,
        block_size: int = 4096,
        vocab_size: int = 65536,
        padding_multiple: int = 4096,
        tie_embeddings: bool = True,
        intermediate_size: int = 17920,
        bias: bool = False,
        architecture_class_name: str = "RecurrentGPT",
        block_class_name: str = "SandwichBlock",
        norm_class_name: str = "RMSNorm_llama",
        norm_eps: float = 0.000001,
        mlp_class_name: str = "GatedMLP",
        nonlin_name: str = "SiLU",
        init_strategy: str = "takase",
        init_orthogonal: bool = False,
        state_init: str = "like-init",
        injection_type: str = "linear",
        n_layers_in_recurrent_block: int = 4,
        mean_recurrence: int = 32,
        sampling_scheme: str = "poisson-lognormal-filling",
        mean_backprop_depth: int = 8,
        n_layers_in_prelude: int = 2,
        n_layers_in_coda: int = 2,
        qk_bias: bool = True,
        activation_checkpoint_impl: str = "per-iteration",
        rope_base: float = 50_000,
        torch_dtype: str | None = "bfloat16",
        transformers_version: str | None = None,
        max_position_embeddings: int | None = None,
        rope_theta: float | None = None,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        padded_vocab_size: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        effective_expected_depth: int | None = None,
        test_time_noise: int | float | None = None,
        test_time_noise_type: str | None = None,
        init_values: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.padded_vocab_size = (
            vocab_size if padded_vocab_size is None else padded_vocab_size
        )
        self.padding_multiple = padding_multiple
        self.tie_embeddings = tie_embeddings
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.architecture_class_name = architecture_class_name
        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.norm_eps = norm_eps
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name
        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.state_init = state_init
        self.injection_type = injection_type
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.mean_recurrence = mean_recurrence
        self.sampling_scheme = sampling_scheme
        self.mean_backprop_depth = mean_backprop_depth
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_coda = n_layers_in_coda
        self.qk_bias = qk_bias
        self.activation_checkpoint_impl = activation_checkpoint_impl
        self.rope_base = rope_base

        # Transformers 5 normalizes RoPE before assigning generic kwargs. These
        # attributes must therefore exist before calling the base initializer.
        self.max_position_embeddings = (block_size if max_position_embeddings is None else max_position_embeddings)
        normalized_rope = dict(rope_parameters or rope_scaling or {})
        resolved_rope_theta = normalized_rope.get("rope_theta")
        if resolved_rope_theta is None:
            resolved_rope_theta = (rope_base if rope_theta is None else rope_theta)
        normalized_rope["rope_theta"] = resolved_rope_theta
        self.rope_theta = resolved_rope_theta
        self.rope_parameters = normalized_rope

        self.test_time_noise = 0 if test_time_noise is None else test_time_noise
        self.test_time_noise_type = ("fixed" if test_time_noise_type is None else test_time_noise_type)

        self.num_key_value_heads = (n_heads if num_key_value_heads is None else num_key_value_heads)
        self.num_attention_heads = n_heads
        self.head_dim = n_embd // n_heads if head_dim is None else head_dim

        if effective_expected_depth is None:
            self.effective_expected_depth = (
                self.n_layers_in_prelude
                + self.n_layers_in_coda
                + self.n_layers_in_recurrent_block * self.mean_recurrence
            )
        else:
            self.effective_expected_depth = effective_expected_depth
        if init_values is None:
            std = sqrt(2 / (5 * self.n_embd))
            self.init_values = {
                "std": std,
                "out_proj": std / sqrt(2 * self.effective_expected_depth),
                "embedding": std,
                "embed_scale": sqrt(self.n_embd),
            }
        else:
            self.init_values = init_values

        dtype = kwargs.pop("dtype", None)
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            tie_word_embeddings=tie_embeddings,
            dtype=torch_dtype if dtype is None else dtype,
            transformers_version=transformers_version,
            **kwargs,
        )


def register_recurrent_configs() -> None:
    """Register every recurrent config required by project entry points."""

    AutoConfig.register(RavenConfig.model_type, RavenConfig, exist_ok=True)


__all__ = [
    "RavenConfig",
    "align_raven_special_token_ids",
    "register_recurrent_configs",
]
