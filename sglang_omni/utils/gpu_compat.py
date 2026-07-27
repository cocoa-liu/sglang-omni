# SPDX-License-Identifier: Apache-2.0
"""GPU-specific runtime compatibility helpers."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Mapping, MutableMapping, Sequence

from sglang_omni.utils.device import get_device_type

from sglang_omni.utils.gpu_memory import (
    _get_device_handle,
    _shutdown_nvml,
    _try_import_pynvml,
    parse_cuda_visible_devices,
    resolve_visible_device_id,
)

logger = logging.getLogger(__name__)

_FLASHINFER_USE_CUDA_NORM = "FLASHINFER_USE_CUDA_NORM"


def _get_compute_capability(
    logical_gpu_id: int,
    env: Mapping[str, str] | None = None,
) -> tuple[int, int] | None:
    """Return CUDA compute capability or None. NPU always returns None (no SM concept)."""
    if get_device_type() == "npu":
        return None

    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(source_env.get("CUDA_VISIBLE_DEVICES"))
    try:
        device_id = resolve_visible_device_id(logical_gpu_id, visible_devices)
    except Exception:
        return None

    pynvml = _try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if visible_devices:
                handle = _get_device_handle(pynvml, device_id)
            else:
                handle = pynvml.nvmlDeviceGetHandleByIndex(logical_gpu_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return int(major), int(minor)
        except Exception as exc:
            logger.debug(
                "NVML compute capability query failed for gpu_id=%s: %s",
                logical_gpu_id,
                exc,
            )
        finally:
            _shutdown_nvml(pynvml)

    if source_env.get("CUDA_VISIBLE_DEVICES") != os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(logical_gpu_id)
            return int(properties.major), int(properties.minor)
    except Exception as exc:
        logger.debug(
            "PyTorch compute capability query failed for gpu_id=%s: %s",
            logical_gpu_id,
            exc,
        )
    return None


def _get_cuda_device_count() -> int | None:
    """Return device count. NPU uses torch.npu.device_count()."""
    if get_device_type() == "npu":
        try:
            torch = importlib.import_module("torch")
            return int(torch.npu.device_count())
        except Exception:
            return None

    pynvml = _try_import_pynvml()
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            return int(pynvml.nvmlDeviceGetCount())
        except Exception as exc:
            logger.debug("NVML device count query failed: %s", exc)
        finally:
            _shutdown_nvml(pynvml)

    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception as exc:
        logger.debug("PyTorch CUDA device count query failed: %s", exc)
    return None


def _visible_gpu_ids(env: Mapping[str, str] | None = None) -> list[int]:
    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(source_env.get("CUDA_VISIBLE_DEVICES"))
    if visible_devices:
        return list(range(len(visible_devices)))
    device_count = _get_cuda_device_count()
    if device_count is not None:
        return list(range(device_count))
    return [0]


def get_visible_gpu_sm_version(
    logical_gpu_id: int,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Return the CUDA SM version for a logical visible GPU."""
    source_env = os.environ if env is None else env
    capability = _get_compute_capability(logical_gpu_id, source_env)
    if capability is None:
        return None
    major, minor = capability
    return major * 10 + minor


def visible_gpus_need_flashinfer_cuda_norm(
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether any visible CUDA device needs the FlashInfer CUDA norm workaround.

    Always returns False on NPU (FlashInfer is CUDA-only).
    """
    if get_device_type() == "npu":
        return False

    source_env = os.environ if env is None else env
    for gpu_id in _visible_gpu_ids(source_env):
        sm_version = get_visible_gpu_sm_version(gpu_id, source_env)
        if sm_version is not None and sm_version >= 100:
            return True
    return False


def get_gpu_compat_env_defaults(
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return env overrides needed for the current visible GPU topology."""
    source_env = os.environ if env is None else env
    if source_env.get(_FLASHINFER_USE_CUDA_NORM) is not None:
        return {}
    if not visible_gpus_need_flashinfer_cuda_norm(source_env):
        return {}
    return {_FLASHINFER_USE_CUDA_NORM: "1"}


def apply_gpu_compat_env_defaults(
    env: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Apply GPU compatibility env overrides to the current process."""
    target_env = os.environ if env is None else env
    overrides = get_gpu_compat_env_defaults(target_env)
    for key, value in overrides.items():
        target_env[key] = value
        logger.info(f"Applied GPU compatibility env override: {key}={value}")
    return overrides


def gpu_ids_support_p2p_mesh(
    logical_gpu_ids: Sequence[int],
    env: Mapping[str, str] | None = None,
) -> bool | None:
    """Return whether the given logical GPUs form a full peer-to-peer mesh.

    On NPU, always returns None (no NVML P2P API available; use HCCL instead).
    """
    if get_device_type() == "npu":
        return None

    ids = list(dict.fromkeys(int(g) for g in logical_gpu_ids))
    if len(ids) < 2:
        return None

    pynvml = _try_import_pynvml()
    if pynvml is None:
        return None

    get_status = getattr(pynvml, "nvmlDeviceGetP2PStatus", None)
    if get_status is None:
        return None
    status_ok = getattr(pynvml, "NVML_P2P_STATUS_OK", 0)
    read_index = getattr(pynvml, "NVML_P2P_CAPS_INDEX_READ", 0)
    # note (luojiaxuan): nvidia-ml-py 13.595.45 ships a stray trailing comma
    # (`= 0,`), making this constant a 1-tuple;
    # nvmlDeviceGetP2PStatus needs a plain int.
    if isinstance(read_index, tuple):
        read_index = read_index[0]

    source_env = os.environ if env is None else env
    visible_devices = parse_cuda_visible_devices(source_env.get("CUDA_VISIBLE_DEVICES"))

    try:
        pynvml.nvmlInit()
        handles = []
        for logical_id in ids:
            device_id = resolve_visible_device_id(logical_id, visible_devices)
            handles.append(_get_device_handle(pynvml, device_id))
        for i, handle_i in enumerate(handles):
            for j, handle_j in enumerate(handles):
                if i == j:
                    continue
                if get_status(handle_i, handle_j, read_index) != status_ok:
                    return False
        return True
    except Exception as exc:
        logger.warning(
            f"NVML P2P mesh query failed for gpus={ids}: {exc}; keeping custom all-reduce disabled",
        )
        return None
    finally:
        _shutdown_nvml(pynvml)


def should_disable_custom_all_reduce_for_gpus(
    logical_gpu_ids: Sequence[int] | None,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Whether to disable SGLang custom all-reduce for a TP thinker.

    On NPU, always returns True (disable) — custom all-reduce is CUDA-only;
    HCCL should be used instead.
    """
    if get_device_type() == "npu":
        return True

    if not logical_gpu_ids:
        return True
    return gpu_ids_support_p2p_mesh(logical_gpu_ids, env) is not True
