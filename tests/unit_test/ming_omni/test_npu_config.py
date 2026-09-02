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
            {"max_running_requests": 16},
        ),
        (
            _FakePlatform("npu", npu=True),
            "npu",
            {
                "max_running_requests": 16,
                "cuda_graph_backend_decode": "full",
                "cuda_graph_max_bs_decode": 16,
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
        assert thinker_args["server_args_overrides"] == expected_overrides

    for stages, talker_name in (
        ({stage.name: stage for stage in configs[1].stages}, "talker"),
        ({stage.name: stage for stage in configs[2].stages}, "talker_stream"),
    ):
        talker_args = stages[talker_name].factory_args
        assert talker_args["device"] == expected_device
        assert talker_args["enable_cuda_graph"] is True


def test_npu_graph_cap_follows_thinker_concurrency(monkeypatch) -> None:
    monkeypatch.setattr(
        ming_config, "current_platform", _FakePlatform("npu", npu=True)
    )

    factory_args = ming_config._thinker_factory_args(max_running_requests=7)
    overrides = factory_args["server_args_overrides"]

    assert overrides["max_running_requests"] == 7
    assert overrides["cuda_graph_max_bs_decode"] == 7
    assert "cuda_graph_bs_decode" not in overrides
