# SPDX-License-Identifier: Apache-2.0
"""Unit contracts for the Ming Talker eager execution path."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from sglang_omni.models.ming_omni.talker.configuration_bailing_talker import (
    MingOmniTalkerConfig,
)
from sglang_omni.models.ming_omni.talker.device_runtime import TalkerDeviceRuntime
from sglang_omni.models.ming_omni.talker.modeling_ming_omni_talker import (
    _load_local_audio,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TALKER_SOURCE = (
    _REPO_ROOT
    / "sglang_omni"
    / "models"
    / "ming_omni"
    / "talker"
    / "modeling_ming_omni_talker.py"
)


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_cfm_eager_path_is_separate_from_cuda_graph_apis() -> None:
    tree = ast.parse(_TALKER_SOURCE.read_text())
    executor = _class_node(tree, "CFMGraphExecutor")
    eager = _method_node(executor, "_execute_eager")

    attributes = [
        node.attr for node in ast.walk(eager) if isinstance(node, ast.Attribute)
    ]
    calls = [
        node.func.attr
        for node in ast.walk(eager)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]

    assert "sample" in calls
    assert "softmax" in calls
    assert "cuda" not in attributes
    assert "CUDAGraph" not in attributes


def test_decode_eager_path_updates_the_supplied_kv_cache() -> None:
    tree = ast.parse(_TALKER_SOURCE.read_text())
    talker = _class_node(tree, "MingOmniTalker")
    decode = _method_node(talker, "_model_decode_forward")
    source = ast.unparse(decode)

    assert "past_key_values=past_key_values" in source
    assert "use_cache=True" in source
    assert "torch.cuda" not in source


def test_generate_normalizes_tensor_cache_length_before_arange() -> None:
    tree = ast.parse(_TALKER_SOURCE.read_text())
    talker = _class_node(tree, "MingOmniTalker")
    generate = _method_node(talker, "generate")
    source = ast.unparse(generate)

    assert "past_seen_tokens = int(past_seen_tokens.item())" in source


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
