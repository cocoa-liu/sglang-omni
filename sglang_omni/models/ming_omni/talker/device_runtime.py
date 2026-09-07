# SPDX-License-Identifier: Apache-2.0
"""Device runtime boundary for Ming-Omni talker worker threads."""

from __future__ import annotations

from contextlib import nullcontext

import torch


class TalkerDeviceRuntime:
    """Provide accelerator stream and graph operations without hard-coding APIs."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.device_module = (
            None if self.device.type == "cpu" else torch.get_device_module(self.device)
        )

    def new_stream(self):
        if self.device_module is None:
            return None
        return self.device_module.Stream(device=self.device)

    def stream_context(self, stream):
        if self.device_module is None:
            return nullcontext()
        return self.device_module.stream(stream)

    def new_graph(self):
        if self.device_module is None:
            raise RuntimeError("device graphs are unavailable on CPU")
        graph_type = getattr(self.device_module, "CUDAGraph", None) or getattr(
            self.device_module, "NPUGraph", None
        )
        if graph_type is None:
            raise RuntimeError(
                f"device graphs are unavailable for {self.device.type!r}"
            )
        return graph_type()

    def graph_context(self, graph):
        if self.device_module is None:
            raise RuntimeError("device graphs are unavailable on CPU")
        return self.device_module.graph(graph, capture_error_mode="thread_local")

    def synchronize(self) -> None:
        if self.device_module is None:
            return
        self.device_module.current_stream(self.device).synchronize()
