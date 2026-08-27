# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for Ming talker graph capture."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch

from sglang_omni.models.ming_omni.talker import (
    modeling_ming_omni_talker as talker_model,
)


def test_cfm_graph_capture_uses_device_runtime(monkeypatch) -> None:
    events: list[object] = []
    graph = object()

    class _Runtime:
        def __init__(self, device):
            events.append(("runtime", device))

        def new_graph(self):
            events.append("new_graph")
            return graph

        @contextmanager
        def graph_context(self, captured_graph):
            events.append(("capture", captured_graph))
            yield

    class _CFM:
        def sample(self, _hidden, _history, noise, *_args, **_kwargs):
            return noise + 1

    monkeypatch.setattr(talker_model, "TalkerDeviceRuntime", _Runtime)
    executor = talker_model.CFMGraphExecutor(
        SimpleNamespace(steps=2, patch_size=2),
        _CFM(),
        lambda latents: latents + 2,
        lambda hidden: torch.stack((hidden[:, 0], hidden[:, 0] + 1), dim=-1),
    )
    input_tensor = torch.randn(1, 1, 4)
    history = torch.randn(1, 2, 4)
    noise = torch.randn(1, 2, 4)
    sde_noise = torch.randn(2, 1, 2, 4)

    executor._initialize_graph(input_tensor, history, noise, sde_noise)

    assert executor.initialized is True
    assert executor.graph is graph
    assert events == [
        ("runtime", input_tensor.device),
        "new_graph",
        ("capture", graph),
    ]
