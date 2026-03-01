from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

MaturityState = Literal["experimental", "supported", "deprecated"]


@dataclass(frozen=True)
class ModelCapability:
    """Capability metadata for a model architecture."""

    architecture: str
    model_family: str
    algorithms: dict[str, tuple[str, ...]]
    required_flags: dict[str, Any] = field(default_factory=dict)
    required_extras: tuple[str, ...] = field(default_factory=tuple)
    maturity: MaturityState = "supported"
    notes: str = ""


DEFAULT_BACKEND_ORDER: dict[str, tuple[str, ...]] = {
    "sft": ("instructlab-training", "mini-trainer"),
    "osft": ("mini-trainer",),
    "lora_sft": ("unsloth",),
}


MODEL_CAPABILITIES: dict[str, ModelCapability] = {
    "Mistral3ForConditionalGeneration": ModelCapability(
        architecture="Mistral3ForConditionalGeneration",
        model_family="mistral",
        algorithms={
            "sft": ("mini-trainer",),
            "osft": ("mini-trainer",),
            "lora_sft": tuple(),
        },
        required_flags={"trust_remote_code": True},
        required_extras=("mistral3",),
        maturity="experimental",
        notes=(
            "Mistral 3 support requires `trust_remote_code=True` and currently uses the "
            "mini-trainer backend path for SFT/OSFT."
        ),
    ),
}


def _looks_like_architecture(value: str) -> bool:
    return value.endswith(("ForCausalLM", "ForConditionalGeneration", "ForSeq2SeqLM"))


def resolve_architecture(
    model_path_or_architecture: Optional[str],
    trust_remote_code: bool | None = None,
) -> Optional[str]:
    """Resolve a model architecture from either an architecture name or model path."""
    if not model_path_or_architecture:
        return None

    if _looks_like_architecture(model_path_or_architecture):
        return model_path_or_architecture

    try:
        from transformers import AutoConfig
    except ImportError:
        return None

    config_kwargs: dict[str, Any] = {}
    if trust_remote_code is not None:
        config_kwargs["trust_remote_code"] = trust_remote_code

    try:
        config = AutoConfig.from_pretrained(model_path_or_architecture, **config_kwargs)
    except Exception:
        return None

    architectures = getattr(config, "architectures", None)
    if architectures:
        return architectures[0]

    config_name = config.__class__.__name__
    if config_name.endswith("Config"):
        return f"{config_name[:-6]}ForCausalLM"
    return None


def get_model_capability(
    model_path_or_architecture: Optional[str],
    trust_remote_code: bool | None = None,
) -> tuple[Optional[str], Optional[ModelCapability]]:
    """Resolve architecture and return matching capability metadata if available."""
    architecture = resolve_architecture(
        model_path_or_architecture=model_path_or_architecture,
        trust_remote_code=trust_remote_code,
    )
    capability = MODEL_CAPABILITIES.get(architecture) if architecture else None
    return architecture, capability


def _format_requirement_help(capability: ModelCapability) -> str:
    messages: list[str] = []

    if capability.required_flags.get("trust_remote_code"):
        messages.append("set `trust_remote_code=True`")

    if capability.required_extras:
        extras = ", ".join(capability.required_extras)
        messages.append(f"install optional extra(s): [{extras}]")

    if not messages:
        return ""

    return " Requirements: " + "; ".join(messages) + "."


def _validate_required_flags(capability: Optional[ModelCapability], trust_remote_code: bool | None) -> None:
    if not capability:
        return

    if capability.required_flags.get("trust_remote_code") and trust_remote_code is not True:
        raise ValueError(
            f"Model architecture '{capability.architecture}' requires `trust_remote_code=True`."
            + _format_requirement_help(capability)
        )


def list_backends_for_model(
    model_path_or_architecture: str,
    algorithm: str,
    trust_remote_code: bool | None = None,
) -> list[str]:
    """List preferred backends for an algorithm and model."""
    _, capability = get_model_capability(model_path_or_architecture, trust_remote_code=trust_remote_code)
    if capability and algorithm in capability.algorithms:
        return list(capability.algorithms[algorithm])
    return list(DEFAULT_BACKEND_ORDER.get(algorithm, tuple()))


def list_model_capabilities(
    model_path_or_architecture: str,
    algorithm: Optional[str] = None,
    trust_remote_code: bool | None = None,
) -> dict[str, Any]:
    """Return capability metadata for a model path or architecture."""
    architecture, capability = get_model_capability(
        model_path_or_architecture=model_path_or_architecture,
        trust_remote_code=trust_remote_code,
    )

    if not capability:
        backends = (
            list(DEFAULT_BACKEND_ORDER.get(algorithm, tuple()))
            if algorithm
            else {k: list(v) for k, v in DEFAULT_BACKEND_ORDER.items()}
        )
        return {
            "model": model_path_or_architecture,
            "architecture": architecture,
            "known": False,
            "maturity": "unknown",
            "recommended_backends": backends,
            "notes": "No explicit compatibility entry found. Using default backend ordering.",
        }

    algorithm_backends: dict[str, list[str]]
    if algorithm:
        algorithm_backends = {algorithm: list(capability.algorithms.get(algorithm, tuple()))}
    else:
        algorithm_backends = {k: list(v) for k, v in capability.algorithms.items()}

    return {
        "model": model_path_or_architecture,
        "architecture": architecture,
        "known": True,
        "model_family": capability.model_family,
        "maturity": capability.maturity,
        "required_flags": dict(capability.required_flags),
        "required_extras": list(capability.required_extras),
        "algorithms": algorithm_backends,
        "notes": capability.notes,
    }


def resolve_backend_selection(
    algorithm_name: str,
    backend_name: str | None,
    available_backends: list[str],
    model_path_or_architecture: Optional[str] = None,
    trust_remote_code: bool | None = None,
) -> str:
    """Resolve and validate backend selection, including `backend='auto'`."""
    architecture, capability = get_model_capability(
        model_path_or_architecture=model_path_or_architecture,
        trust_remote_code=trust_remote_code,
    )

    capability_backends = capability.algorithms.get(algorithm_name) if capability else None
    default_backends = DEFAULT_BACKEND_ORDER.get(algorithm_name, tuple(available_backends))

    if backend_name not in (None, "auto"):
        if capability and capability_backends is not None and backend_name not in capability_backends:
            requirement_help = _format_requirement_help(capability)
            raise ValueError(
                f"Backend '{backend_name}' is not compatible with algorithm '{algorithm_name}' "
                f"for architecture '{architecture}'. Supported backend(s): {list(capability_backends)}."
                f" Capability state: {capability.maturity}.{requirement_help}"
            )

        if backend_name not in available_backends:
            raise ValueError(
                f"Backend '{backend_name}' not found for algorithm '{algorithm_name}'. "
                f"Available backend(s): {available_backends}"
            )

        _validate_required_flags(capability, trust_remote_code=trust_remote_code)
        return backend_name

    preferred_backends = list(capability_backends) if capability_backends is not None else list(default_backends)

    if capability and capability_backends is not None and len(capability_backends) == 0:
        requirement_help = _format_requirement_help(capability)
        raise ValueError(
            f"Algorithm '{algorithm_name}' is not currently supported for architecture '{architecture}'. "
            f"Capability state: {capability.maturity}.{requirement_help}"
        )

    for candidate in preferred_backends:
        if candidate in available_backends:
            _validate_required_flags(capability, trust_remote_code=trust_remote_code)
            return candidate

    if not available_backends:
        raise ValueError(f"No backends available for algorithm '{algorithm_name}'")

    if capability and capability_backends is not None:
        requirement_help = _format_requirement_help(capability)
        raise ValueError(
            f"No compatible backend is currently registered for algorithm '{algorithm_name}' "
            f"and architecture '{architecture}'. Preferred backend(s): {preferred_backends}; "
            f"registered backend(s): {available_backends}.{requirement_help}"
        )

    return available_backends[0]
