# SPDX-License-Identifier: Apache-2.0
"""Distributed backend policy tests for the Ming image encoder."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from sglang_omni.models.ming_omni.components import image_encoder


@dataclass
class _FakePlatform:
    backend: str

    def get_torch_distributed_backend_str(self) -> str:
        return self.backend


@pytest.mark.parametrize("backend", ["gloo", "nccl", "hccl"])
def test_distributed_backend_comes_from_platform(monkeypatch, backend: str) -> None:
    monkeypatch.setattr(
        image_encoder,
        "current_platform",
        _FakePlatform(backend),
    )

    assert image_encoder._distributed_backend() == backend
