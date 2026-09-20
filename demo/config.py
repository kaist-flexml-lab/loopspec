"""Load and validate the streaming demo configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    name: str
    mode: str
    step_size: int | None = None
    second_step: int | None = None


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    path: str
    chat: bool
    enable_thinking: bool
    configurations: Mapping[str, StrategyConfig]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    ignore_eos: bool


@dataclass(frozen=True, slots=True)
class DemoConfig:
    gpu: str
    memory_fraction: float
    server_seed: int
    generation: GenerationConfig
    models: Mapping[str, ModelConfig]
    active_model: str
    active_configuration: str

    @property
    def model(self) -> ModelConfig:
        return self.models[self.active_model]

    @property
    def strategy(self) -> StrategyConfig:
        return self.model.configurations[self.active_configuration]


def _mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _only_keys(name: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"{name} has unknown keys: {sorted(unknown)}")


def _positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _read_strategy(name: str, raw: Any) -> StrategyConfig:
    values = _mapping(f"configuration {name}", raw)
    _only_keys(
        f"configuration {name}",
        values,
        {
            "mode",
            "step_size",
            "second_step",
        },
    )
    mode = values.get("mode")
    if mode not in {"baseline", "loopspec"}:
        raise ValueError(f"configuration {name}: mode must be baseline or loopspec")

    if mode == "baseline":
        if set(values) != {"mode"}:
            raise ValueError(f"configuration {name}: baseline has extra options")
        return StrategyConfig(name=name, mode=mode)

    allowed = {
        "mode",
        "step_size",
        "second_step",
    }
    if set(values) - allowed:
        raise ValueError(f"configuration {name}: invalid LoopSpec options")
    if "step_size" not in values:
        raise ValueError(f"configuration {name}: LoopSpec requires step_size")
    step_size = _positive_int(
        f"configuration {name}: step_size", values["step_size"]
    )
    second_step = values.get("second_step")
    if second_step is not None:
        second_step = _positive_int(
            f"configuration {name}: second_step", second_step
        )
    return StrategyConfig(
        name=name,
        mode=mode,
        step_size=step_size,
        second_step=second_step,
    )


def _read_model(name: str, raw: Any) -> ModelConfig:
    values = _mapping(f"model {name}", raw)
    _only_keys(
        f"model {name}",
        values,
        {"path", "chat", "enable_thinking", "configs"},
    )
    path = _nonempty_string(f"model {name}: path", values.get("path"))
    chat = values.get("chat", False)
    thinking = values.get("enable_thinking", False)
    if type(chat) is not bool or type(thinking) is not bool:
        raise ValueError(f"model {name}: chat flags must be booleans")
    if thinking and not chat:
        raise ValueError(f"model {name}: enable_thinking requires chat")
    raw_configs = _mapping(f"model {name}: configs", values.get("configs"))
    if not raw_configs:
        raise ValueError(f"model {name}: configs cannot be empty")
    configs = {
        str(config_name): _read_strategy(str(config_name), raw_config)
        for config_name, raw_config in raw_configs.items()
    }
    return ModelConfig(
        name=name,
        path=path,
        chat=chat,
        enable_thinking=thinking,
        configurations=configs,
    )


def load_config(path: str | Path = DEFAULT_CONFIG) -> DemoConfig:
    """Load one demo YAML and validate every advertised model/configuration."""

    path = Path(path)
    with path.open() as file:
        raw = yaml.safe_load(file)
    values = _mapping("demo config", raw)
    _only_keys(
        "demo config", values, {"active", "server", "generation", "models"}
    )

    active = _mapping("active", values.get("active"))
    _only_keys("active", active, {"model", "config"})
    active_model = _nonempty_string("active.model", active.get("model"))
    active_configuration = _nonempty_string(
        "active.config", str(active.get("config", ""))
    )

    server = _mapping("server", values.get("server"))
    _only_keys(
        "server", server, {"gpu", "mem_fraction_static", "random_seed"}
    )
    gpu_value = server.get("gpu", 0)
    if isinstance(gpu_value, bool) or not isinstance(gpu_value, (int, str)):
        raise ValueError("server.gpu must be an integer or string")
    gpu = str(gpu_value)
    if not gpu:
        raise ValueError("server.gpu cannot be empty")
    memory_fraction = _number(
        "server.mem_fraction_static", server.get("mem_fraction_static", 0.4)
    )
    if not 0 < memory_fraction < 1:
        raise ValueError("server.mem_fraction_static must be between 0 and 1")
    server_seed = _nonnegative_int(
        "server.random_seed", server.get("random_seed", 1234)
    )

    generation_values = _mapping("generation", values.get("generation"))
    _only_keys(
        "generation",
        generation_values,
        {"max_new_tokens", "temperature", "top_p", "seed", "ignore_eos"},
    )
    max_new_tokens = _positive_int(
        "generation.max_new_tokens",
        generation_values.get("max_new_tokens", 256),
    )
    temperature = _number(
        "generation.temperature", generation_values.get("temperature", 0)
    )
    if temperature < 0:
        raise ValueError("generation.temperature must be nonnegative")
    top_p = _number("generation.top_p", generation_values.get("top_p", 1))
    if not 0 < top_p <= 1:
        raise ValueError("generation.top_p must be in (0, 1]")
    seed = _nonnegative_int("generation.seed", generation_values.get("seed", 1234))
    ignore_eos = generation_values.get("ignore_eos", False)
    if type(ignore_eos) is not bool:
        raise ValueError("generation.ignore_eos must be a boolean")
    generation = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        ignore_eos=ignore_eos,
    )

    raw_models = _mapping("models", values.get("models"))
    models = {
        str(model_name): _read_model(str(model_name), model_values)
        for model_name, model_values in raw_models.items()
    }
    if active_model not in models:
        raise ValueError(f"unknown active model: {active_model}")
    if active_configuration not in models[active_model].configurations:
        raise ValueError(
            f"model {active_model} has no config {active_configuration}"
        )

    return DemoConfig(
        gpu=gpu,
        memory_fraction=memory_fraction,
        server_seed=server_seed,
        generation=generation,
        models=models,
        active_model=active_model,
        active_configuration=active_configuration,
    )


def server_arguments(config: DemoConfig) -> list[str]:
    """Translate the selected demo profile to the serving CLI."""

    model = config.model
    strategy = config.strategy
    arguments = [
        "--model",
        model.path,
        "--host",
        "0.0.0.0",
        "--port",
        "30000",
        "--mem-fraction-static",
        str(config.memory_fraction),
        "--random-seed",
        str(config.server_seed),
        "--max-running-requests",
        "1",
        "--stream-interval",
        "1",
        "--enable-metrics",
    ]
    if strategy.mode == "loopspec":
        arguments += ["--loopspec", "--step-size", str(strategy.step_size)]
        if strategy.second_step is not None:
            arguments += [
                "--second-step",
                str(strategy.second_step),
            ]
    return arguments
