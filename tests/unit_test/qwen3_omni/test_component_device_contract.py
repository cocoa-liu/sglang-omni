# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from sglang_omni.models.qwen3_omni import stages
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    create_code2wav_scheduler,
    load_code2wav_model,
)
from sglang_omni.models.qwen3_omni.components.common import (
    assert_module_device,
    assert_tensor_tree_device,
    resolve_component_device,
)
from sglang_omni.models.qwen3_omni.components.image_encoder import (
    Qwen3OmniImageEncoder,
)


def test_component_device_requires_explicit_device_or_placement() -> None:
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        resolve_component_device(
            device=None,
            gpu_id=None,
            component="test_component",
        )


def test_explicit_cpu_mode_remains_supported() -> None:
    assert (
        resolve_component_device(
            device="cpu",
            gpu_id=None,
            component="test_component",
        )
        == "cpu"
    )


def test_placement_is_concrete_and_conflicts_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang_omni.utils.device.get_device_string",
        lambda gpu_id: f"cuda:{gpu_id}",
    )
    assert (
        resolve_component_device(
            device=None,
            gpu_id=2,
            component="test_component",
        )
        == "cuda:2"
    )
    with pytest.raises(ValueError, match="conflicts with placement"):
        resolve_component_device(
            device="cpu",
            gpu_id=2,
            component="test_component",
        )


def test_module_and_tensor_device_assertions() -> None:
    module = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    assert_module_device(module, "cpu", component="test_component")
    assert_tensor_tree_device(
        {"x": torch.ones(1), "nested": [torch.zeros(1)]},
        "cpu",
        component="test_component",
        boundary="forward_input",
    )
    with pytest.raises(RuntimeError, match="device placement mismatch"):
        assert_module_device(module, "meta", component="test_component")
    with pytest.raises(RuntimeError, match="forward_output device mismatch"):
        assert_tensor_tree_device(
            torch.ones(1),
            "meta",
            component="test_component",
            boundary="forward_output",
        )


@pytest.mark.parametrize(
    "factory",
    [stages.create_image_encoder_executor, stages.create_audio_encoder_executor],
)
def test_encoder_stage_factories_accept_placement_gpu_id(factory) -> None:
    assert "gpu_id" in inspect.signature(factory).parameters


@pytest.mark.parametrize(
    "factory",
    [Qwen3OmniImageEncoder, Qwen3OmniAudioEncoder, load_code2wav_model],
)
def test_component_entrypoints_fail_before_loading_without_device(factory) -> None:
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        factory("unused-model-path")


def test_code2wav_factory_fails_before_loading_without_device() -> None:
    with pytest.raises(ValueError, match="CPU fallback is disabled"):
        create_code2wav_scheduler("unused-model-path")
