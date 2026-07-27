# SPDX-License-Identifier: Apache-2.0
"""Hardware device abstraction layer for SGLang-Omni.

Provides a unified interface for NPU and CUDA device detection and operations.
All device-specific code paths are centralized here.
"""

from __future__ import annotations

import torch

_device_type: str | None = None


def patch_cuda_lazy_init_for_npu() -> None:
    """Patch torch internals for NPU when running on CPU-only torch + torch_npu.

    torch==2.10.0+cpu + torch_npu has issues that block model weight loading:
    1.  ``tensor.to("npu:0")`` inside ``module._apply(convert)`` triggers
        ``torch.cuda._lazy_init()`` which asserts CUDA compile support.
    2.  The C++ tensor movement to NPU internally hits CUDA code paths that
        fail with RuntimeError on CPU-only torch builds.

    This function patches both layers so model weights can load onto NPU.
    """
    if get_device_type() != "npu":
        return

    import torch.cuda as cuda_mod
    import torch.npu

    # Patch 1: suppress AssertionError in CUDA lazy init
    _orig_lazy = cuda_mod._lazy_init

    def _patched_lazy_init() -> None:
        try:
            _orig_lazy()
        except (AssertionError, RuntimeError):
            pass

    cuda_mod._lazy_init = _patched_lazy_init

    # Patch 2: wrap Module.to() so tensor movement survives C++ CUDA errors
    torch.npu.set_device(0)
    _orig_module_to = torch.nn.Module.to

    def _npu_safe_module_to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = (
            torch._C._nn._parse_to(*args, **kwargs)
        )
        if torch.npu.is_available():
            with torch.no_grad():
                try:
                    self.cast_weight(device)
                except Exception:
                    pass

        def convert(t):
            try:
                if convert_to_format is not None and t.dim() == 4:
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except RuntimeError as exc:
                if "cuda" in str(exc).lower():
                    return t.cpu().to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                    )
                raise

        return self._apply(convert)

    torch.nn.Module.to = _npu_safe_module_to


def patch_inductor_skip_compile_for_npu() -> None:
    """Prevent triton compilation failures from crashing stage processes.

    CANN 9.0.0 BiSheng cannot compile fused kernels the talker produces.
    This monkey-patches the triton compilation so that failures are silently
    swallowed, allowing the inductor to fall back to eager mode.
    """
    if get_device_type() != "npu":
        return

    # Path 1: NPUTritonKernel._precompile_worker
    try:
        import torch_npu._inductor.npu_triton_heuristics as _nth

        _orig_worker = _nth.NPUTritonKernel._precompile_worker

        def _safe_worker(self):
            compile_results = []
            for c in self.configs:
                try:
                    compile_results.append(self._precompile_config(c))
                except Exception:
                    pass
            if len(compile_results) == 0:
                self.compile_results = []
                self.configs = None
                return
            self.compile_results = compile_results
            self.configs = None

        _nth.NPUTritonKernel._precompile_worker = _safe_worker
    except Exception:
        pass

    # Path 2: Intercept triton.compile to patch generated kernel source.
    # The inductor generates .py kernel files with float64 conversions
    # that CANN 9.0.0 BiSheng cannot compile.  We replace float64→float32
    # in the generated source before passing it to triton.compile.
    try:
        import triton

        _orig_triton_compile = triton.compile

        def _patched_triton_compile(src, target=None, options=None, **kwargs):
            # If src is a string (file path) and contains float64, patch it
            if isinstance(src, str) and src.endswith(".py"):
                try:
                    with open(src) as f:
                        code = f.read()
                    if "float64" in code or "tl.float64" in code:
                        code = code.replace("tl.float64", "tl.float32")
                        code = code.replace("float64", "float32")
                        with open(src, "w") as f:
                            f.write(code)
                except Exception:
                    pass
            return _orig_triton_compile(src, target=target, options=options, **kwargs)

        triton.compile = _patched_triton_compile
    except Exception:
        pass


def _try_sglang_npu() -> bool:
    """Check if SGLang reports NPU environment.

    Returns True if sglang is available and is_npu() returns True.
    Returns False if sglang is not installed or is_npu() returns False.
    """
    try:
        from sglang.srt.utils import is_npu

        return is_npu()
    except ImportError:
        return False


def _try_torch_npu() -> bool | None:
    """Check if torch.npu is available.

    Returns True if torch_npu module can be imported and is_available() returns True.
    Returns False if imported but not available.
    Returns None if torch_npu cannot be imported (ModuleNotFoundError).
    """
    try:
        import torch.npu  # noqa: F401

        return torch.npu.is_available()
    except (ImportError, ModuleNotFoundError):
        return None


def get_device_type() -> str:
    """Detect current hardware platform.

    Detection chain: sglang.is_npu() → torch.npu → torch.cuda → "cpu".
    The result is cached — subsequent calls return the same value.

    Returns:
        "npu" for Ascend NPU, "cuda" for NVIDIA GPU, "cpu" for no accelerator.
    """
    global _device_type

    if _device_type is not None:
        return _device_type

    # Step 1: Check SGLang NPU detection
    if _try_sglang_npu():
        _device_type = "npu"
        return _device_type

    # Step 2: Check torch.npu directly
    npu_available = _try_torch_npu()
    if npu_available is not None:
        if npu_available:
            _device_type = "npu"
        else:
            _device_type = "cuda" if torch.cuda.is_available() else "cpu"
        return _device_type

    # Step 3: torch.npu not importable, fall back to torch.cuda
    _device_type = "cuda" if torch.cuda.is_available() else "cpu"
    return _device_type


def get_device_name() -> str:
    """Return device name without index.

    Returns:
        "npu" or "cuda" or "cpu".
    """
    return get_device_type()


def get_device_string(gpu_id: int) -> str:
    """Return standard device string like 'npu:0' or 'cuda:0'.

    Args:
        gpu_id: Non-negative integer GPU/NPU index.

    Returns:
        Formatted device string.

    Raises:
        ValueError: If gpu_id is negative.
    """
    if gpu_id < 0:
        raise ValueError(f"gpu_id must be >= 0, got {gpu_id}")
    return f"{get_device_name()}:{gpu_id}"


def set_device(device: str | torch.device) -> None:
    """Set the current device.

    Args:
        device: Device string (e.g., "npu:0") or torch.device object.

    Raises:
        RuntimeError: If no accelerator device is detected.
    """
    device_type = get_device_type()
    if device_type == "npu":
        torch.npu.set_device(device)
    elif device_type == "cuda":
        torch.cuda.set_device(device)
    else:
        raise RuntimeError(f"无法设置设备: 未检测到可用设备 (当前: {device_type})")


def synchronize(device: str | torch.device) -> None:
    """Synchronize the device stream.

    On NPU, torch.npu.synchronize() takes no device argument.
    On CUDA, torch.cuda.synchronize() is called with the device.

    Args:
        device: Device to synchronize (ignored on NPU path).

    Raises:
        RuntimeError: If no accelerator device is detected.
    """
    device_type = get_device_type()
    if device_type == "npu":
        torch.npu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize(device)
    else:
        raise RuntimeError(f"无法同步: 未检测到可用设备 (当前: {device_type})")


def create_event() -> torch.cuda.Event | torch.npu.Event:
    """Create a device Event for timing/synchronization.

    Returns:
        torch.npu.Event or torch.cuda.Event depending on device type.

    Raises:
        RuntimeError: If no accelerator device is detected.
    """
    device_type = get_device_type()
    if device_type == "npu":
        return torch.npu.Event()
    elif device_type == "cuda":
        return torch.cuda.Event()
    else:
        raise RuntimeError(f"无法创建 Event: 未检测到可用设备 (当前: {device_type})")


def is_available() -> bool:
    """Check if an accelerator device is available.

    Returns:
        True if NPU or CUDA is detected, False for CPU.
    """
    return get_device_type() in ("npu", "cuda")


def device_count() -> int:
    """Return the number of available accelerator devices.

    Returns:
        Number of NPU or CUDA devices. Returns 0 on CPU or if the query fails.
    """
    device_type = get_device_type()
    try:
        if device_type == "npu":
            return torch.npu.device_count()
        elif device_type == "cuda":
            return torch.cuda.device_count()
        else:
            return 0
    except Exception:
        return 0


def get_device_properties(gpu_id: int) -> object:
    """Return device properties for the given GPU/NPU index.

    Args:
        gpu_id: Device index.

    Returns:
        DeviceProperties object from torch.npu or torch.cuda.

    Raises:
        RuntimeError: If no accelerator device or invalid gpu_id.
    """
    device_type = get_device_type()
    if device_type == "npu":
        return torch.npu.get_device_properties(gpu_id)
    elif device_type == "cuda":
        return torch.cuda.get_device_properties(gpu_id)
    else:
        raise RuntimeError(f"无法获取设备属性: 未检测到可用设备 (当前: {device_type})")


def get_device_name_str(gpu_id: int) -> str:
    """Return human-readable device name.

    Args:
        gpu_id: Device index.

    Returns:
        Device name string, e.g., "Ascend910B" or "NVIDIA A100".

    Raises:
        ValueError: If gpu_id is negative.
        RuntimeError: If no accelerator device is detected.
    """
    if gpu_id < 0:
        raise ValueError(f"gpu_id must be >= 0, got {gpu_id}")
    device_type = get_device_type()
    if device_type == "npu":
        return torch.npu.get_device_name(gpu_id)
    elif device_type == "cuda":
        return torch.cuda.get_device_name(gpu_id)
    else:
        raise RuntimeError(f"无法获取设备名称: 未检测到可用设备 (当前: {device_type})")


def get_distributed_backend() -> str:
    """Return the torch.distributed backend name for the current device.

    Returns:
        "hccl" for NPU, "nccl" for CUDA.

    Raises:
        RuntimeError: If no accelerator device is detected.
    """
    device_type = get_device_type()
    if device_type == "npu":
        return "hccl"
    elif device_type == "cuda":
        return "nccl"
    else:
        raise RuntimeError(f"无法确定分布式后端: 未检测到支持的设备 (当前: {device_type})")


def get_visible_devices_env() -> str:
    """Return the environment variable name for device visibility.

    Returns:
        "ASCEND_RT_VISIBLE_DEVICES" for NPU, "CUDA_VISIBLE_DEVICES" for CUDA.
    """
    if get_device_type() == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"
