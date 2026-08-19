# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni components."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang_omni.utils import load_hf_config

logger = logging.getLogger(__name__)


def load_thinker_config(model_path: str) -> Any:
    cfg = load_hf_config(model_path, trust_remote_code=True, local_files_only=True)
    return cfg.thinker_config


def resolve_component_device(
    *,
    device: str | torch.device | None,
    gpu_id: int | None,
    component: str,
) -> str:
    """Resolve one concrete device without silently falling back to CPU."""
    placement = None
    if gpu_id is not None:
        from sglang_omni.utils.device import get_device_string

        placement = torch.device(get_device_string(int(gpu_id)))

    if device is None:
        if placement is None:
            raise ValueError(
                f"{component} requires an explicit device or placement gpu_id; "
                "CPU fallback is disabled"
            )
        return str(placement)

    requested = torch.device(device)
    if placement is not None:
        same_type = requested.type == placement.type
        same_index = requested.index in (None, placement.index)
        if not (same_type and same_index):
            raise ValueError(
                f"{component} device {requested} conflicts with placement "
                f"device {placement}"
            )
        return str(placement)

    if requested.type in ("cuda", "npu") and requested.index is None:
        raise ValueError(
            f"{component} accelerator device must include an index when no "
            "placement gpu_id is available"
        )
    return str(requested)


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    return actual.type == expected.type and (
        expected.index is None or actual.index == expected.index
    )


def assert_module_device(
    module: nn.Module,
    expected_device: str | torch.device,
    *,
    component: str,
) -> None:
    """Fail startup when an inference parameter or buffer is misplaced."""
    expected = torch.device(expected_device)
    wrong: list[str] = []
    parameter_bytes = 0
    buffer_bytes = 0
    parameter_count = 0
    buffer_count = 0
    for name, value in module.named_parameters():
        parameter_count += 1
        parameter_bytes += value.numel() * value.element_size()
        if not _device_matches(value.device, expected):
            wrong.append(f"parameter:{name}={value.device}")
    for name, value in module.named_buffers():
        buffer_count += 1
        buffer_bytes += value.numel() * value.element_size()
        if not _device_matches(value.device, expected):
            wrong.append(f"buffer:{name}={value.device}")
    if wrong:
        raise RuntimeError(
            f"{component} device placement mismatch: expected={expected}, "
            f"wrong={wrong[:20]}"
        )
    logger.info(
        "component_module_device_verified component=%s device=%s "
        "parameters=%d buffers=%d parameter_bytes=%d buffer_bytes=%d",
        component,
        expected,
        parameter_count,
        buffer_count,
        parameter_bytes,
        buffer_bytes,
    )


def assert_tensor_tree_device(
    value: Any,
    expected_device: str | torch.device,
    *,
    component: str,
    boundary: str,
) -> None:
    """Fail a representative forward when tensor inputs or outputs leave device."""
    expected = torch.device(expected_device)
    wrong: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            if not _device_matches(item.device, expected):
                wrong.append(f"{path}={item.device}")
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, boundary)
    if wrong:
        raise RuntimeError(
            f"{component} {boundary} device mismatch: expected={expected}, "
            f"wrong={wrong[:20]}"
        )


@dataclass(frozen=True)
class Qwen3OmniSpec:
    """Lightweight spec extracted from the HF config."""

    model_path: str
    audio_token_id: int
    image_token_id: int
    spatial_merge_size: int

    @classmethod
    def from_model_path(cls, model_path: str) -> "Qwen3OmniSpec":
        thinker_cfg = load_thinker_config(model_path)
        vision_cfg = thinker_cfg.vision_config
        return cls(
            model_path=model_path,
            audio_token_id=int(thinker_cfg.audio_token_id),
            image_token_id=int(thinker_cfg.image_token_id),
            spatial_merge_size=int(vision_cfg.spatial_merge_size),
        )
