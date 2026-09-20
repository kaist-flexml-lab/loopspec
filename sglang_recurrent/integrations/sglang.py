"""Install recurrent architecture selection in SGLang's config parser."""

from __future__ import annotations

import json
from copy import copy
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer
from transformers import PreTrainedConfig, PreTrainedTokenizerBase

from .huggingface import (
    align_raven_special_token_ids,
    register_recurrent_configs,
)


def use_raven_tokenizer_special_tokens(
    model_config: Any,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Align a Raven ModelConfig with the loaded tokenizer's BOS/EOS/PAD IDs.

    Updates the owned HF configs and SGLang's EOS set in place; returns None.
    Does nothing for other model families. Raven requires a tokenizer and
    raises ValueError if it is missing, matching the server's CLI contract.
    """

    config = model_config.hf_config
    if config.model_type != "huginn_raven":
        return
    if tokenizer is None:
        raise ValueError("Raven special-token alignment requires the target tokenizer")

    # Keep distinct text/generation configs consistent with the main HF config.
    mirrors = []
    text_config = getattr(model_config, "hf_text_config", None)
    if text_config is not None and text_config is not config:
        mirrors.append(text_config)
    generation_config = getattr(model_config, "hf_generation_config", None)
    if generation_config is not None:
        # Replace the generation config with a copy before changing its IDs.
        generation_config = copy(generation_config)
        model_config.hf_generation_config = generation_config
        mirrors.append(generation_config)
    align_raven_special_token_ids(config, tokenizer, *mirrors)

    # SGLang also stores stop-token IDs separately from the HF configs.
    eos_token_id = tokenizer.eos_token_id
    model_config.hf_eos_token_id = (set() if eos_token_id is None else {eos_token_id})


def use_ouro_tokenizer_special_tokens(
    model_config: Any,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Keep Ouro's thinking delimiters in decoded generated text.

    Accepts an SGLang ModelConfig, a bare HF config, or the detokenizer's
    SimpleNamespace carrying model_type, together with the loaded tokenizer.
    Updates the tokenizer in place and returns None; other model families are
    ignored. A missing tokenizer is allowed for Ouro baseline's token-ID-only
    mode (--skip-tokenizer-init), where there is no text decoding to patch.

    Text decoding requires a native tokenizers.Tokenizer backend for all Ouro
    1.4B/2.6B variants, including Thinking. Other backends raise TypeError.

    Ouro checkpoints currently mark ``<think>`` and ``</think>`` as special
    tokens even though they are ordinary generation delimiters.  SGLang's
    default decode path uses ``skip_special_tokens=True``, so patch the loaded
    fast-tokenizer backend before it is used by the scheduler.  Rebuilding the
    backend from its JSON is intentional: changing the Python-side decoder
    entries alone does not update tokenizers' internal special-token registry.
    """

    config = getattr(model_config, "hf_config", model_config)
    if getattr(config, "model_type", None) != "ouro" or tokenizer is None:
        return

    backend = getattr(tokenizer, "_tokenizer", None)
    if not isinstance(backend, Tokenizer):
        raise TypeError(
            "Ouro text decoding requires a tokenizers.Tokenizer backend; "
            "use --tokenizer-backend=huggingface --tokenizer-mode=auto"
        )
    state = json.loads(backend.to_str())
    for added_token in state.get("added_tokens", ()):
        if added_token.get("content") in {"<think>", "</think>"}:
            added_token["special"] = False
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(state))


def use_recurrent_tokenizer_special_tokens(
    model_config: Any,
    tokenizer: PreTrainedTokenizerBase | None,
) -> None:
    """Apply family-specific fixes to an SGLang config and its tokenizer.

    Updates the supplied objects in place and returns None. Each hook checks
    the model family itself. A missing tokenizer raises for Raven but is
    allowed for Ouro's token-ID-only baseline mode.
    """

    use_raven_tokenizer_special_tokens(model_config, tokenizer)
    use_ouro_tokenizer_special_tokens(model_config, tokenizer)


_PARSER_FOR_ARCHITECTURE = {
    "LoopedRecurrentServingModel": "looped_transformer_baseline",
    "RecurrentServingModel": "looped_transformer_loopspec",
}
_sglang_config_installed = False


def install_sglang_config_compatibility(architecture: str) -> str:
    """Register an SGLang parser for one recurrent serving architecture.

    ``architecture`` selects ``LoopedRecurrentServingModel`` (baseline) or
    ``RecurrentServingModel`` (LoopSpec). Returns the corresponding parser name
    for SGLang's ``model_config_parser`` option; unknown names raise ValueError.

    Config loading remains owned by SGLang's built-in ``hf`` parser. Restricting
    the architecture override to recurrent target types leaves non-recurrent
    configs untouched if they share the same parser.
    """

    global _sglang_config_installed
    # SGLang's HTTP parent parses the config before the scheduler imports the
    # model implementation. Register recurrent configs explicitly in every
    # process instead of relying on a modeling import side effect or a released
    # checkpoint's Transformers 4-era remote config.
    register_recurrent_configs()
    try:
        selected_parser = _PARSER_FOR_ARCHITECTURE[architecture]
    except KeyError as error:
        raise ValueError(
            f"unsupported recurrent serving architecture: {architecture!r}"
        ) from error
    if _sglang_config_installed:
        return selected_parser

    # Defer SGLang imports: even importing its package can initialize CUDA.
    from sglang.srt.configs.model_config_parser_registry import (
        ModelConfigParserBase,
        get_model_config_parser,
        register_model_config_parser,
    )
    # Importing the config loader registers SGLang's built-in parsers.
    from sglang.srt.utils.hf_transformers import config as _config  # noqa: F401

    hf_parser = get_model_config_parser("hf")

    def register(parser_name: str, serving_architecture: str) -> None:
        """Bind a parser name to a serving architecture; return None.

        Each call captures its own architecture for the parser registered below.
        """

        @register_model_config_parser(parser_name)
        class LoopedTransformerConfigParser(ModelConfigParserBase):
            def parse(
                self: LoopedTransformerConfigParser,
                model: str | Path,
                trust_remote_code: bool,
                revision: str | None = None,
                **kwargs: Any,
            ) -> PreTrainedConfig:
                """Load an HF config and select the recurrent serving class.

                ``model`` is a Hub ID or local path. Remote-code permission,
                revision, and extra loader options pass through to the HF parser.
                Returns its config, overriding architectures for Raven/Ouro.
                """

                config = hf_parser.parse(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
                if config.model_type in {"huginn_raven", "ouro"}:
                    config.architectures = [serving_architecture]
                return config

    # Install both parsers once; callers receive the name they requested.
    for serving_architecture, parser_name in _PARSER_FOR_ARCHITECTURE.items():
        register(parser_name, serving_architecture)

    _sglang_config_installed = True
    return selected_parser


def register_recurrent_models() -> None:
    """Register recurrent model classes in SGLang's current process.

    Takes no arguments and returns None. Scans the modeling package for model
    entry points, overwrites existing registrations, and propagates load errors.
    """

    # Importing the registry scans SGLang's models; do it only when requested.
    from sglang.srt.models.registry import ModelRegistry

    ModelRegistry.register(
        "sglang_recurrent.modeling",
        overwrite=True,
        strict=True,
    )


__all__ = [
    "install_sglang_config_compatibility",
    "register_recurrent_models",
    "use_ouro_tokenizer_special_tokens",
    "use_raven_tokenizer_special_tokens",
    "use_recurrent_tokenizer_special_tokens",
]
