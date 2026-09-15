# SPDX-License-Identifier: Apache-2.0
"""Opt-in hardware smoke checks; no model downloads or CUDA fallback."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNI_RUN_NPU_TESTS") != "1",
    reason="Set OMNI_RUN_NPU_TESTS=1 on a reserved NPU",
)


def test_npu_runtime():
    import torch
    import torch_npu  # noqa: F401

    from sglang_omni.platforms import current_platform

    assert current_platform.is_npu(), "NPU CI must not fall back to CPU/CUDA"
    assert torch.npu.is_available()
    assert torch.npu.device_count() == 1, "Reserve exactly one visible NPU"
    print(f"Device: {torch.npu.get_device_name(0)}")
    # Exercise an actual device kernel, not just package imports.
    value = torch.empty((16, 16), dtype=torch.bfloat16, device="npu")
    value.fill_(1)
    actual = value @ value
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), torch.full((16, 16), 16.0).bfloat16())


def test_npu_graph_replay():
    import torch
    import torch_npu  # noqa: F401

    from sglang_omni.platforms import current_platform

    value = torch.ones(16, device="npu")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            result = value + 1
    torch.npu.current_stream().wait_stream(stream)
    backend = current_platform.get_device_graph_backend(value.device)
    assert backend is not None
    with backend.capture(stream=stream) as graph:
        result = value + 1
    value.fill_(3)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), torch.full((16,), 4.0))
