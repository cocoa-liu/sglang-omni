# SPDX-License-Identifier: Apache-2.0
"""Device runtime boundary for Ming-Omni talker worker threads."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class TalkerDeviceRuntime:
    """Provide accelerator stream operations without hard-coding CUDA/NPU."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.module = (
            None if self.device.type == "cpu" else torch.get_device_module(self.device)
        )

    def new_stream(self):
        if self.module is None:
            return None
        return self.module.Stream(device=self.device)

    def stream_context(self, stream):
        if self.module is None:
            return nullcontext()
        return self.module.stream(stream)

    def synchronize(self) -> None:
        if self.module is None:
            return
        self.module.current_stream(self.device).synchronize()
