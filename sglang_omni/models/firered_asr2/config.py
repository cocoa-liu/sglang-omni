# SPDX-License-Identifier: Apache-2.0
"""Single-stage native FireRedASR2-LLM transcription pipeline."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    AudioChunkingConfig,
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)


class FireRedASR2PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "FireRedASR2ForConditionalGeneration"
    allow_audio_chunking: ClassVar[bool] = True
    max_native_clip_s: ClassVar[float] = 40.0
    audio_chunking: AudioChunkingConfig = AudioChunkingConfig(max_audio_clip_s=30.0)
    model_path: str
    entry_stage: str = "asr"
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": EngineStageConfig,
    }
    stages: list[StageConfig] = Field(
        default_factory=lambda: [
            EngineStageConfig(
                name="asr",
                process="asr",
                factory_path="sglang_omni.models.firered_asr2.stages.create_firered_asr2_executor",
                factory=FactoryArgs(dtype="bfloat16", max_new_tokens=256),
                engine=EngineArgs(max_running_requests=8),
                gpu=0,
                terminal=True,
            )
        ]
    )


EntryClass = FireRedASR2PipelineConfig
