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
        (
            _FakePlatform("cuda"),
            "cuda",
            {},
        ),
        (
            _FakePlatform("npu", npu=True),
            "npu",
            {
                "cuda_graph_backend_decode": "full",
            },
        ),
    ],
)
def test_ming_pipelines_use_platform_device_and_graph_policy(
    monkeypatch,
    platform: _FakePlatform,
    expected_device: str,
    expected_overrides: dict[str, object],
) -> None:
    monkeypatch.setattr(ming_config, "current_platform", platform)

    configs = [
        ming_config.MingOmniPipelineConfig(model_path="dummy"),
        ming_config.MingOmniSpeechPipelineConfig(model_path="dummy"),
        ming_config.MingOmniStreamingSpeechPipelineConfig(model_path="dummy"),
    ]
    for config in configs:
        stages = {stage.name: stage for stage in config.stages}
        assert stages["audio_encoder"].factory_args["device"] == expected_device
        assert stages["image_encoder"].factory_args["device"] == expected_device

        thinker_args = stages["thinker"].factory_args
        if expected_overrides:
            assert thinker_args["server_args_overrides"] == expected_overrides
        else:
            assert "server_args_overrides" not in thinker_args

    for stages, talker_name in (
        ({stage.name: stage for stage in configs[1].stages}, "talker"),
        ({stage.name: stage for stage in configs[2].stages}, "talker_stream"),
    ):
        talker_args = stages[talker_name].factory_args
        assert talker_args["device"] == expected_device
