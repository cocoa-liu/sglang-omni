# SPDX-License-Identifier: Apache-2.0
"""Stage factory for native FireRedASR2-LLM inference."""


from collections.abc import Mapping

from sglang_omni.models.firered_asr2.engine_builder import FireRedASR2EngineBuilder
from sglang_omni.models.firered_asr2.request_builders import FireRedRequestData
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def create_firered_asr2_executor(
    model_path: str,
    *,
    mm_embedding_cache_size_bytes: int,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 256,
    max_running_requests: int = 8,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[FireRedRequestData]:

    return FireRedASR2EngineBuilder(
        max_new_tokens=max_new_tokens,
        max_running_requests=max_running_requests,
        mm_embedding_cache_size_bytes=mm_embedding_cache_size_bytes,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )
