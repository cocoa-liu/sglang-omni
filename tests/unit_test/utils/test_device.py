# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sglang_omni.utils.device — hardware device abstraction layer.

[no integration test] — pure functions with mockable external dependencies (torch.npu, torch.cuda, sglang).
Real-hardware verification is in Feature 6 (FR-007) end-to-end manual test.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_device_cache():
    """Reset module-level device_type cache before each test for isolation."""
    # Clear any pre-cached state
    import importlib

    if "sglang_omni.utils.device" in importlib.sys.modules:
        mod = importlib.sys.modules["sglang_omni.utils.device"]
        mod._device_type = None


# ── FUNC/happy — get_device_type ──────────────────────────────────


def test_get_device_type_npu_via_sglang(reset_device_cache):
    """A1: SGLang is_npu() returns True → 'npu' (mocked _try_sglang_npu)."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=True):
        result = dev_mod.get_device_type()
        assert result == "npu"


def test_get_device_type_npu_direct(reset_device_cache):
    """A1 alt: torch.npu.is_available() returns True → 'npu'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.is_available.return_value = True
            result = dev_mod.get_device_type()
            assert result == "npu", f"Expected 'npu', got '{result}'"
            assert dev_mod._device_type == "npu"


def test_get_device_type_cuda(reset_device_cache):
    """A2: torch.npu unavailable, torch.cuda available → 'cuda'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(dev_mod, "_try_torch_npu", return_value=None):
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                result = dev_mod.get_device_type()
                assert result == "cuda"


def test_get_device_type_cpu(reset_device_cache):
    """A3: No devices available → 'cpu'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(dev_mod, "_try_torch_npu", return_value=None):
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                result = dev_mod.get_device_type()
                assert result == "cpu"


def test_get_device_type_npu_import_fails_no_cuda(reset_device_cache):
    """A4: torch.npu import fails, no CUDA → 'cpu'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(dev_mod, "_try_torch_npu", return_value=None):
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                result = dev_mod.get_device_type()
                assert result == "cpu"


# ── FUNC/happy — get_device_string ─────────────────────────────


def test_get_device_string_npu0(reset_device_cache):
    """A5: NPU → 'npu:0'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    result = dev_mod.get_device_string(0)
    assert result == "npu:0"


def test_get_device_string_cuda0(reset_device_cache):
    """A6: CUDA → 'cuda:0'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    result = dev_mod.get_device_string(0)
    assert result == "cuda:0"


# ── FUNC/happy — get_distributed_backend ──────────────────────


def test_get_distributed_backend_npu(reset_device_cache):
    """A7: NPU → 'hccl'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    result = dev_mod.get_distributed_backend()
    assert result == "hccl"


def test_get_distributed_backend_cuda(reset_device_cache):
    """A8: CUDA → 'nccl'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    result = dev_mod.get_distributed_backend()
    assert result == "nccl"


# ── FUNC/happy — create_event ──────────────────────────────────


def test_create_event_npu(reset_device_cache):
    """A9: NPU → torch.npu.Event."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    mock_event = mock.MagicMock()
    with mock.patch.object(torch, "npu", create=True) as mock_npu:
        mock_npu.Event.return_value = mock_event
        result = dev_mod.create_event()
        assert result is mock_event


def test_create_event_cuda(reset_device_cache):
    """A20/C2: CUDA → torch.cuda.Event."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    mock_event = mock.MagicMock()
    with mock.patch.object(torch.cuda, "Event", return_value=mock_event):
        result = dev_mod.create_event()
        assert result is mock_event


# ── FUNC/happy — get_device_name ───────────────────────────────


def test_get_device_name_npu(reset_device_cache):
    """A10: NPU → 'npu'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    assert dev_mod.get_device_name() == "npu"


def test_get_device_name_cuda(reset_device_cache):
    """C3: CUDA → 'cuda'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    assert dev_mod.get_device_name() == "cuda"


# ── FUNC/happy — is_available ──────────────────────────────────


def test_is_available_npu(reset_device_cache):
    """A11: NPU → True."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    assert dev_mod.is_available() is True


def test_is_available_cpu(reset_device_cache):
    """A12: CPU → False."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cpu"
    assert dev_mod.is_available() is False


# ── FUNC/happy — device_count ──────────────────────────────────


def test_device_count_cuda(reset_device_cache):
    """A13: CUDA → delegates to torch.cuda.device_count()."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    with mock.patch.object(torch.cuda, "device_count", return_value=4):
        assert dev_mod.device_count() == 4


# ── FUNC/happy — get_device_properties ────────────────────────


def test_get_device_properties_npu(reset_device_cache):
    """A14: NPU → delegates to torch.npu.get_device_properties()."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    mock_props = mock.MagicMock()
    mock_props.name = "Ascend910B"
    with mock.patch.object(torch, "npu", create=True) as mock_npu:
        mock_npu.get_device_properties.return_value = mock_props
        result = dev_mod.get_device_properties(0)
        assert result.name == "Ascend910B"
        mock_npu.get_device_properties.assert_called_once_with(0)


# ── FUNC/happy — get_device_name_str ──────────────────────────


def test_get_device_name_str_cuda(reset_device_cache):
    """A15: CUDA → delegates to torch.cuda.get_device_name()."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    with mock.patch.object(torch.cuda, "get_device_name", return_value="NVIDIA A100"):
        assert dev_mod.get_device_name_str(0) == "NVIDIA A100"


# ── FUNC/happy — get_visible_devices_env ──────────────────────


def test_get_visible_devices_env_npu(reset_device_cache):
    """A16: NPU → ASCEND_RT_VISIBLE_DEVICES."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    assert dev_mod.get_visible_devices_env() == "ASCEND_RT_VISIBLE_DEVICES"


def test_get_visible_devices_env_cuda(reset_device_cache):
    """A17: CUDA → CUDA_VISIBLE_DEVICES."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    assert dev_mod.get_visible_devices_env() == "CUDA_VISIBLE_DEVICES"


# ── FUNC/happy — set_device ────────────────────────────────────


def test_set_device_npu(reset_device_cache):
    """A18: NPU → calls torch.npu.set_device()."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    with mock.patch.object(torch, "npu", create=True) as mock_npu:
        dev_mod.set_device("npu:0")
        mock_npu.set_device.assert_called_once_with("npu:0")


def test_set_device_cuda_with_torch_device(reset_device_cache):
    """C4: CUDA with torch.device → calls torch.cuda.set_device()."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    dev = torch.device("cuda:0")
    with mock.patch.object(torch.cuda, "set_device") as mock_set:
        dev_mod.set_device(dev)
        mock_set.assert_called_once_with(dev)


# ── FUNC/happy — synchronize ──────────────────────────────────


def test_synchronize_cuda(reset_device_cache):
    """A19: CUDA → calls torch.cuda.synchronize(device)."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    dev = torch.device("cuda:0")
    with mock.patch.object(torch.cuda, "synchronize") as mock_sync:
        dev_mod.synchronize(dev)
        mock_sync.assert_called_once_with(dev)


# ── BNDRY/edge — SGLang fallback ─────────────────────────────


def test_get_device_type_sglang_import_error_fallback(reset_device_cache):
    """B1: SGLang ImportError → falls back to torch.npu."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.is_available.return_value = True
            result = dev_mod.get_device_type()
            assert result == "npu"


def test_get_device_type_all_unavailable(reset_device_cache):
    """B2: SGLang returns False, torch.npu unavailable, torch.cuda unavailable → 'cpu'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(dev_mod, "_try_torch_npu", return_value=None):
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                result = dev_mod.get_device_type()
                assert result == "cpu"


# ── BNDRY/edge — synchronize NPU no-arg ──────────────────────


def test_synchronize_npu_no_device_arg(reset_device_cache):
    """B3: NPU → calls torch.npu.synchronize() WITHOUT device argument."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    with mock.patch.object(torch, "npu", create=True) as mock_npu:
        dev_mod.synchronize("npu:0")
        # Must call synchronize() with zero arguments on NPU
        mock_npu.synchronize.assert_called_once_with()


# ── BNDRY/edge — cache ───────────────────────────────────────


def test_get_device_type_cache(reset_device_cache):
    """B4: Second call returns cached value, not re-detected."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = None
    with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=False):
        with mock.patch.object(torch, "npu", create=True) as mock_npu:
            mock_npu.is_available.return_value = True
            result1 = dev_mod.get_device_type()
            assert result1 == "npu"

            # Now change mock so NPU would NOT be detected
            mock_npu.is_available.return_value = False
            result2 = dev_mod.get_device_type()
            # Must return cached value, not re-detect
            assert result2 == "npu"


# ── BNDRY/edge — get_device_string boundaries ────────────────


def test_get_device_string_negative_gpu_id(reset_device_cache):
    """B5: gpu_id=-1 → ValueError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    with pytest.raises(ValueError, match="gpu_id must be >= 0"):
        dev_mod.get_device_string(-1)


def test_get_device_string_zero_boundary(reset_device_cache):
    """B6: gpu_id=0 valid boundary → 'npu:0'."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    result = dev_mod.get_device_string(0)
    assert result == "npu:0"
    assert result.endswith(":0")


def test_get_device_string_large_gpu_id(reset_device_cache):
    """C1: gpu_id=7 → 'npu:7' (no upper-bound validation in this function)."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    result = dev_mod.get_device_string(7)
    assert result == "npu:7"


# ── BNDRY/error — cpu device raises RuntimeError ──────────────


def test_set_device_cpu_raises(reset_device_cache):
    """B7: CPU → RuntimeError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cpu"
    with pytest.raises(RuntimeError, match="未检测到可用设备"):
        dev_mod.set_device("cpu")


def test_create_event_cpu_raises(reset_device_cache):
    """B8: CPU → RuntimeError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cpu"
    with pytest.raises(RuntimeError, match="无法创建 Event"):
        dev_mod.create_event()


def test_synchronize_cpu_raises(reset_device_cache):
    """B9: CPU → RuntimeError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cpu"
    with pytest.raises(RuntimeError, match="无法同步"):
        dev_mod.synchronize("cpu")


def test_get_distributed_backend_cpu_raises(reset_device_cache):
    """B10: CPU → RuntimeError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cpu"
    with pytest.raises(RuntimeError, match="无法确定分布式后端"):
        dev_mod.get_distributed_backend()


# ── BNDRY/error — device_count error path ────────────────────


def test_device_count_driver_error_graceful(reset_device_cache):
    """B11: NPU device_count() driver error → returns 0."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    with mock.patch.object(torch, "npu", create=True) as mock_npu:
        mock_npu.device_count.side_effect = RuntimeError("driver error")
        result = dev_mod.device_count()
        assert result == 0


# ── BNDRY/error — get_device_properties out of range ─────────


def test_get_device_properties_out_of_range(reset_device_cache):
    """B12: CUDA, gpu_id=99 → torch RuntimeError propagates."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    with mock.patch.object(torch.cuda, "device_count", return_value=2):
        with mock.patch.object(
            torch.cuda, "get_device_properties", side_effect=RuntimeError("invalid device")
        ):
            with pytest.raises(RuntimeError, match="invalid device"):
                dev_mod.get_device_properties(99)


# ── BNDRY/error — get_device_name_str negative gpu_id ────────


def test_get_device_name_str_negative_gpu_id(reset_device_cache):
    """B13: NPU, gpu_id=-1 → ValueError."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "npu"
    with pytest.raises(ValueError, match="gpu_id must be >= 0"):
        dev_mod.get_device_name_str(-1)


# ── IAPI-002 — Event type check ──────────────────────────────


def test_create_event_cuda_duck_type(reset_device_cache):
    """C2: CUDA event is Event-like."""
    import sglang_omni.utils.device as dev_mod

    dev_mod._device_type = "cuda"
    mock_event = mock.MagicMock(spec=torch.cuda.Event)
    with mock.patch.object(torch.cuda, "Event", return_value=mock_event):
        result = dev_mod.create_event()
        # Duck-type check: should behave like an Event
        assert hasattr(result, "record")
        assert hasattr(result, "wait")
