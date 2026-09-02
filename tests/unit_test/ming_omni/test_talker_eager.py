# SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the Ming Talker eager execution path."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
)
from sglang_omni.models.ming_omni.talker.device_runtime import TalkerDeviceRuntime
from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
    CFMGraphExecutor,
    MingOmniTalker,
)


def test_cfm_eager_path_executes_model_components() -> None:
    calls: dict[str, object] = {}

    class _CFM:
        def sample(self, *args, abort_event=None):
            calls["sample"] = (args, abort_event)
            return args[2] + 1

    def aggregator(latents):
        calls["aggregator"] = latents
        return latents + 2

    def stop_head(hidden):
        calls["stop_head"] = hidden
        return torch.tensor([[0.0, 1.0]], dtype=hidden.dtype)

    executor = CFMGraphExecutor(
        SimpleNamespace(patch_size=2, steps=3),
        _CFM(),
        aggregator,
        stop_head,
        enable_cuda_graph=False,
    )
    input_tensor = torch.randn(1, 1, 4)
    his_lat = torch.randn(1, 2, 4)

    gen_lat, inputs_embeds, stop_out = executor.execute(input_tensor, his_lat)

    assert "sample" in calls
    assert calls["aggregator"] is gen_lat
    torch.testing.assert_close(inputs_embeds, gen_lat + 2)
    torch.testing.assert_close(stop_out.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(calls["stop_head"], input_tensor[:, -1, :])
    assert executor.initialized is False


def test_decode_forward_passes_cache_to_model() -> None:
    calls: dict[str, object] = {}
    expected = object()

    class _Model:
        def __call__(self, **kwargs):
            calls.update(kwargs)
            return expected

    talker = SimpleNamespace(model=_Model())
    embeddings = torch.randn(1, 1, 4)
    cache_position = torch.tensor([3])
    cache = object()

    result = MingOmniTalker._model_decode_forward(
        talker,
        inputs_embeds=embeddings,
        cache_position=cache_position,
        past_key_values=cache,
    )

    assert result is expected
    assert calls["inputs_embeds"] is embeddings
    assert calls["cache_position"] is cache_position
    assert calls["past_key_values"] is cache
    assert calls["use_cache"] is True
    assert calls["output_hidden_states"] is True


def test_use_torch_attention_overrides_both_talker_backends() -> None:
    config = object.__new__(MingOmniTalkerConfig)
    config.flowmodel = {"attn_backend": "flash_attn"}
    config.aggregator = {"attn_backend": "flash_attn"}

    config.use_torch_attention()

    assert config.flowmodel["attn_backend"] == "torch"
    assert config.aggregator["attn_backend"] == "torch"


def test_accelerator_device_runtime_delegates_stream_and_graph(monkeypatch) -> None:
    stream = object()
    graph = object()
    synchronize = Mock()
    module = SimpleNamespace(
        Stream=Mock(return_value=stream),
        stream=Mock(return_value=nullcontext()),
        current_stream=Mock(return_value=SimpleNamespace(synchronize=synchronize)),
        NPUGraph=Mock(return_value=graph),
        graph=Mock(return_value=nullcontext()),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: module)

    runtime = TalkerDeviceRuntime("npu:2")
    with runtime.stream_context(runtime.new_stream()):
        pass
    runtime.synchronize()
    with runtime.graph_context(runtime.new_graph()):
        pass

    device = torch.device("npu:2")
    module.Stream.assert_called_once_with(device=device)
    module.stream.assert_called_once_with(stream)
    module.current_stream.assert_called_once_with(device)
    synchronize.assert_called_once_with()
    module.NPUGraph.assert_called_once_with()
    module.graph.assert_called_once_with(graph, capture_error_mode="thread_local")
