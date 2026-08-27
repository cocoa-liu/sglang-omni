# SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the Ming Talker eager execution path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
)
from sglang_omni.models.ming_omni.talker.device_runtime import TalkerDeviceRuntime
from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
    CFMGraphExecutor,
    MingOmniTalker,
    _load_local_audio,
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


def test_cpu_device_runtime_uses_noop_stream_context() -> None:
    runtime = TalkerDeviceRuntime("cpu")

    assert runtime.new_stream() is None
    with runtime.stream_context(None):
        pass
    runtime.synchronize()


def test_accelerator_device_runtime_delegates_stream_operations(monkeypatch) -> None:
    events: list[object] = []
    stream = object()

    class _StreamContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    module = SimpleNamespace(
        Stream=lambda *, device: events.append(("new", device)) or stream,
        stream=lambda value: events.append(("context", value)) or _StreamContext(),
        current_stream=lambda device: SimpleNamespace(
            synchronize=lambda: events.append(("sync", device))
        ),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: module)

    runtime = TalkerDeviceRuntime("npu:2")
    created = runtime.new_stream()
    with runtime.stream_context(created):
        events.append("body")
    runtime.synchronize()

    assert events == [
        ("new", torch.device("npu:2")),
        ("context", stream),
        "enter",
        "body",
        "exit",
        ("sync", torch.device("npu:2")),
    ]


def test_accelerator_device_runtime_delegates_graph_operations(monkeypatch) -> None:
    events: list[object] = []
    graph = object()

    class _GraphContext:
        def __enter__(self):
            events.append("capture_enter")

        def __exit__(self, *_args):
            events.append("capture_exit")

    module = SimpleNamespace(
        NPUGraph=lambda: events.append("new_graph") or graph,
        graph=lambda value, **kwargs: events.append(("graph", value, kwargs))
        or _GraphContext(),
    )
    monkeypatch.setattr(torch, "get_device_module", lambda _device: module)

    runtime = TalkerDeviceRuntime("npu:2")
    created = runtime.new_graph()
    with runtime.graph_context(created):
        events.append("body")

    assert events == [
        "new_graph",
        ("graph", graph, {"capture_error_mode": "thread_local"}),
        "capture_enter",
        "body",
        "capture_exit",
    ]


def test_local_wav_loader_does_not_require_torchcodec(tmp_path: Path) -> None:
    path = tmp_path / "voice.wav"
    expected = torch.linspace(-0.5, 0.5, 1600).numpy()
    sf.write(path, expected, 16000, subtype="FLOAT")

    waveform, sample_rate = _load_local_audio(str(path))

    assert sample_rate == 16000
    assert waveform.shape == (1, 1600)
    torch.testing.assert_close(waveform[0], torch.from_numpy(expected))
