import sys
import types

import pytest

# `training_hub.__init__` imports memory estimator symbols that reference
# `mini_trainer.osft_utils`. Stub this optional module for unit-test isolation.
mini_trainer_stub = types.ModuleType("mini_trainer")
mini_trainer_osft_utils_stub = types.ModuleType("mini_trainer.osft_utils")
mini_trainer_osft_utils_stub.MODEL_CONFIGS = {}
mini_trainer_stub.osft_utils = mini_trainer_osft_utils_stub
sys.modules.setdefault("mini_trainer", mini_trainer_stub)
sys.modules.setdefault("mini_trainer.osft_utils", mini_trainer_osft_utils_stub)

from training_hub.algorithms import create_algorithm
from training_hub.model_capabilities import (
    list_backends_for_model,
    list_model_capabilities,
    resolve_backend_selection,
)
from training_hub.algorithms.sft import InstructLabTrainingSFTBackend, MiniTrainerSFTBackend


def test_list_model_capabilities_for_mistral3_architecture():
    caps = list_model_capabilities("Mistral3ForConditionalGeneration")

    assert caps["known"] is True
    assert caps["architecture"] == "Mistral3ForConditionalGeneration"
    assert caps["maturity"] == "experimental"
    assert caps["algorithms"]["sft"] == ["mini-trainer"]
    assert caps["required_flags"]["trust_remote_code"] is True


def test_list_backends_for_model_prefers_minitrainer_for_mistral3_sft():
    backends = list_backends_for_model(
        "Mistral3ForConditionalGeneration",
        algorithm="sft",
    )

    assert backends == ["mini-trainer"]


def test_resolve_backend_selection_requires_trust_remote_code_for_mistral3():
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        resolve_backend_selection(
            algorithm_name="sft",
            backend_name="auto",
            available_backends=["instructlab-training", "mini-trainer"],
            model_path_or_architecture="Mistral3ForConditionalGeneration",
            trust_remote_code=False,
        )


def test_resolve_backend_selection_rejects_unsupported_manual_backend_for_mistral3_lora():
    with pytest.raises(ValueError, match="not compatible"):
        resolve_backend_selection(
            algorithm_name="lora_sft",
            backend_name="unsloth",
            available_backends=["unsloth"],
            model_path_or_architecture="Mistral3ForConditionalGeneration",
            trust_remote_code=True,
        )


def test_create_algorithm_auto_routes_mistral3_sft_to_minitrainer():
    algo = create_algorithm(
        "sft",
        backend_name="auto",
        model_path_or_architecture="Mistral3ForConditionalGeneration",
        trust_remote_code=True,
    )

    assert isinstance(algo.backend, MiniTrainerSFTBackend)


def test_create_algorithm_auto_routes_unknown_sft_to_default_backend():
    algo = create_algorithm(
        "sft",
        backend_name="auto",
        model_path_or_architecture="LlamaForCausalLM",
    )

    assert isinstance(algo.backend, InstructLabTrainingSFTBackend)


def test_create_algorithm_rejects_manual_incompatible_backend_for_mistral3_sft():
    with pytest.raises(ValueError, match=r"Supported backend\(s\): \['mini-trainer'\]"):
        create_algorithm(
            "sft",
            backend_name="instructlab-training",
            model_path_or_architecture="Mistral3ForConditionalGeneration",
            trust_remote_code=True,
        )
