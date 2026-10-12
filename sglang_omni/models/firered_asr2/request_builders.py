# SPDX-License-Identifier: Apache-2.0
"""Transcription requests lowered into native SGLang multimodal requests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.firered_asr2.audio import (
    FireRedFeatureExtractor,
    num_audio_tokens,
)
from sglang_omni.preprocessing.transcription import prepare_audio
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.token_text_streaming import (
    make_token_text_stream_output_builder,
)
from sglang_omni.scheduling.types import StreamOutputBuilder

SPEECH_TOKEN = "<speech>"
PROMPT_PREFIX = "<|im_start|>user\n"
PROMPT_SUFFIX = "请转写音频为文字<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class FireRedRequestData(SGLangARRequestData):
    output_ids: list[int] | None = None
    audio_duration_s: float = 0.0
    engine_start_s: float = 0.0


class FireRedRequestBuilder(Protocol):
    def __call__(self, payload: StagePayload) -> FireRedRequestData: ...


class FireRedResultAdapter(Protocol):
    def __call__(self, data: FireRedRequestData) -> StagePayload: ...


def build_prompt_ids(
    tokenizer: PreTrainedTokenizerBase, count: int
) -> tuple[list[int], int, int]:
    if count < 1:
        raise ValueError("Audio must produce at least one token")
    else:
        pass
    before = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
    after = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
    speech_id = tokenizer.convert_tokens_to_ids(SPEECH_TOKEN)
    if speech_id is None or speech_id == tokenizer.unk_token_id:
        raise ValueError("Checkpoint tokenizer is missing the <speech> token")
    else:
        pass
    return before + [speech_id] * count + after, len(before), speech_id


def make_firered_adapters(
    *,
    tokenizer: PreTrainedTokenizerBase,
    feature_extractor: FireRedFeatureExtractor,
    max_new_tokens: int = 256,
    downsample_rate: int = 2,
) -> tuple[FireRedRequestBuilder, FireRedResultAdapter]:
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def request_builder(payload: StagePayload) -> FireRedRequestData:
        params = payload.request.params or {}
        audio = prepare_audio(
            payload, source_name="FireRedASR2-LLM", max_duration_s=40.0
        )
        feature = feature_extractor(audio.waveform)
        count = num_audio_tokens(feature.shape[0], downsample_rate)
        ids, offset, speech_id = build_prompt_ids(tokenizer, count)
        item = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=audio.fingerprint_int,
            feature=feature,
            model_specific_data={"num_audio_tokens": count},
        )
        item.set_pad_value()
        ids = [item.pad_value if token == speech_id else token for token in ids]
        item.offsets = [(offset, offset + count - 1)]
        mm = MultimodalInputs(mm_items=[item], num_image_tokens=count)
        mm.audio_token_id = speech_id
        requested_limit = params.get("max_new_tokens")
        limit = int(max_new_tokens if requested_limit is None else requested_limit)
        if limit < 1:
            raise ValueError("max_new_tokens must be positive")
        else:
            pass
        temperature = float(params.get("temperature") or 0.0)
        sampling = SamplingParams(
            max_new_tokens=limit,
            temperature=temperature,
            top_p=float(params.get("top_p", 1.0)),
            repetition_penalty=float(params.get("repetition_penalty", 1.0)),
            stop_token_ids=[int(eos_id)],
        )
        sampling.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=ids,
            sampling_params=sampling,
            vocab_size=len(tokenizer),
            extra_key=audio.fingerprint,
        )
        req.multimodal_inputs = mm
        req._codec_suppress_tokens = (
            None  # noqa: leading-underscore  # Shared runner contract.
        )
        return FireRedRequestData(
            input_ids=torch.tensor(ids, dtype=torch.long),
            req=req,
            max_new_tokens=limit,
            temperature=temperature,
            audio_duration_s=audio.duration_s,
            engine_start_s=time.perf_counter(),
            stage_payload=payload,
        )

    def result_adapter(data: FireRedRequestData) -> StagePayload:
        payload = data.stage_payload
        text = tokenizer.decode(data.output_ids or [], skip_special_tokens=True).strip()
        elapsed = time.perf_counter() - data.engine_start_s
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "text": text,
                "modality": "text",
                "duration_s": data.audio_duration_s,
                "asr_latency_s": elapsed,
                "usage": {"engine_time_s": elapsed},
            },
        )

    return request_builder, result_adapter


def make_firered_stream_builder(
    tokenizer: PreTrainedTokenizerBase,
) -> StreamOutputBuilder[FireRedRequestData]:
    return make_token_text_stream_output_builder(
        decode_fn=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        build_message_data=lambda delta: {"text": delta, "modality": "text"},
        build_message_metadata=lambda token: {"token_id": token},
        pending_ids_attr="firered_pending_text_ids",
        last_emit_attr="firered_last_text_emit",
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        allow_terminal_flush=True,
    )
