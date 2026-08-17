# SPDX-License-Identifier: Apache-2.0
"""Platform policy tests for the Ming-Omni text pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sglang_omni.models.ming_omni import config as ming_config


@dataclass
class _FakePlatform:
    device_type: str
    npu: bool = False

    def is_npu(self) -> bool:
        return self.npu


@pytest.mark.parametrize(
    ("platform", "expected_device", "expected_overrides"),
    [
        (_FakePlatform("cuda"), "cuda", {}),
        (
            _FakePlatform("npu", npu=True),
            "npu",
            {"disable_cuda_graph": True},
        ),
    ],
)
def test_text_pipeline_uses_platform_device_and_thinker_graph_policy(
    monkeypatch,
    platform: _FakePlatform,
    expected_device: str,
    expected_overrides: dict[str, bool],
) -> None:
    monkeypatch.setattr(ming_config, "current_platform", platform)

    config = ming_config.MingOmniPipelineConfig(model_path="dummy")
    stages = {stage.name: stage for stage in config.stages}

    assert stages["audio_encoder"].factory_args["device"] == expected_device
    assert stages["image_encoder"].factory_args["device"] == expected_device
    thinker_args = stages["thinker"].factory_args
    if expected_overrides:
        assert thinker_args["server_args_overrides"] == expected_overrides
    else:
        assert "server_args_overrides" not in thinker_args


def test_npu_graph_policy_is_shared_by_streaming_thinker(monkeypatch) -> None:
    monkeypatch.setattr(
        ming_config,
        "current_platform",
        _FakePlatform("npu", npu=True),
    )

    config = ming_config.MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    thinker = next(stage for stage in config.stages if stage.name == "thinker")

    assert thinker.factory_args["server_args_overrides"] == {"disable_cuda_graph": True}


@pytest.mark.parametrize(
    ("platform", "expected_device", "expected_graph"),
    [
        (_FakePlatform("cuda"), "cuda", True),
        (_FakePlatform("npu", npu=True), "npu", False),
    ],
)
def test_non_streaming_talker_uses_platform_graph_policy(
    monkeypatch,
    platform: _FakePlatform,
    expected_device: str,
    expected_graph: bool,
) -> None:
    monkeypatch.setattr(ming_config, "current_platform", platform)

    config = ming_config.MingOmniSpeechPipelineConfig(model_path="dummy")
    talker = next(stage for stage in config.stages if stage.name == "talker")

    assert talker.factory_args["device"] == expected_device
    assert talker.factory_args["enable_cuda_graph"] is expected_graph


@pytest.mark.parametrize(
    ("platform", "expected_device", "expected_graph"),
    [
        (_FakePlatform("cuda"), "cuda", True),
        (_FakePlatform("npu", npu=True), "npu", False),
    ],
)
def test_streaming_talker_uses_platform_graph_policy(
    monkeypatch,
    platform: _FakePlatform,
    expected_device: str,
    expected_graph: bool,
) -> None:
    monkeypatch.setattr(ming_config, "current_platform", platform)

    config = ming_config.MingOmniStreamingSpeechPipelineConfig(model_path="dummy")
    talker = next(stage for stage in config.stages if stage.name == "talker_stream")

    assert talker.factory_args["device"] == expected_device
    assert talker.factory_args["enable_cuda_graph"] is expected_graph
