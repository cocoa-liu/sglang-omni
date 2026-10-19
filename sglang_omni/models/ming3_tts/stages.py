# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Ming 3 text-to-speech."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal, TypedDict

import onnxruntime
import torch
from pydantic import BaseModel, ConfigDict, JsonValue
from safetensors import safe_open
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from transformers import AutoConfig, PreTrainedTokenizerBase

from sglang_omni.models.ming3_tts.model_runner import (
    Ming3TTSModelRunner,
    Ming3TTSRequestData,
    StopPolicy,
)
from sglang_omni.models.ming3_tts.sglang_model import (
    VOCODER_WEIGHT_PREFIX,
    Ming3TTSConfig,
    Ming3TTSModel,
)
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.audio_features import cached_fbank
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

INSTRUCTION_PREFIX = (
    "<role>HUMAN</role>Please generate speech based on the following description."
    "\n  speaker_1:"
)
TEXT_PREFIX = "\n Text input:\n"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
SPEAKER_ENCODER_FILE = "campplus.onnx"
SPEAKER_SAMPLE_RATE = 16000
SPEAKER_MEL_BINS = 80
SPEAKER_ENCODER_THREADS = 2


class SpeechConfig(BaseModel):
    """The tts_config subset this pipeline implements; other contracts fail at load."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    prompt_format: Literal["legacy_antmmf"]
    with_rml: Literal[True]
    conditioning_mode: Literal["speaker_only"]
    decode_alignment: Literal["old_shift"]
    patch_size: Literal[1]
    history_size: Literal[1]
    latent_dim: int
    speaker_embedding_dim: int
    sampling_temperature: float
    stop_threshold: float
    minimum_decode_steps: int
    maximum_decode_steps: int
    audio_patch_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    speaker_start_token_id: int
    speaker_end_token_id: int


class SpeechPrompt(TypedDict):
    input_ids: torch.Tensor
    speaker_position: int
    speaker_embedding: torch.Tensor


def load_speech_config(checkpoint: Path) -> SpeechConfig:
    configuration = json.loads((checkpoint / "config.json").read_text())
    return SpeechConfig.model_validate(configuration["tts_config"])


def build_speech_prompt(
    tokenizer: PreTrainedTokenizerBase, speech_config: SpeechConfig, text: str
) -> tuple[list[int], int]:
    """Return the prompt token ids and the index the speaker embedding replaces."""

    def encode(segment: str) -> list[int]:
        return tokenizer.encode(segment, add_special_tokens=False)

    prefix_ids = encode(INSTRUCTION_PREFIX) + [speech_config.speaker_start_token_id]
    input_ids = (
        prefix_ids
        + [speech_config.audio_patch_token_id, speech_config.speaker_end_token_id]
        + encode(TEXT_PREFIX)
        + encode(text)
        + encode(ASSISTANT_PREFIX)
        + [speech_config.audio_start_token_id]
    )
    return input_ids, len(prefix_ids)


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    checkpoint = Path(resolve_checkpoint(model_path))
    speech_config = load_speech_config(checkpoint)
    tokenizer = get_tokenizer(str(checkpoint), trust_remote_code=False)
    session_options = onnxruntime.SessionOptions()
    session_options.intra_op_num_threads = SPEAKER_ENCODER_THREADS
    speaker_encoder = onnxruntime.InferenceSession(
        str(checkpoint / SPEAKER_ENCODER_FILE),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    speaker_input_name = speaker_encoder.get_inputs()[0].name

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        if isinstance(inputs, str):
            text, references = inputs, []
        else:
            text, references = inputs["text"], inputs.get("references", [])
        if not text.strip():
            raise ValueError("Ming 3 TTS requires non-empty input text")
        elif len(references) > 1:
            raise ValueError("Ming 3 TTS accepts at most one reference audio")
        else:
            pass
        if not references:
            speaker_embedding = torch.zeros(speech_config.speaker_embedding_dim)
        else:
            reference = references[0]
            if "data" in reference:
                reference_source = base64.b64decode(reference["data"], validate=True)
            else:
                reference_source = reference["audio_path"]
            waveform = torch.from_numpy(
                load_audio(reference_source, target_sample_rate=SPEAKER_SAMPLE_RATE)
            )
            filterbank = cached_fbank(
                waveform[None],
                num_mel_bins=SPEAKER_MEL_BINS,
                sample_frequency=SPEAKER_SAMPLE_RATE,
            )
            filterbank = filterbank - filterbank.mean(dim=0, keepdim=True)
            speaker_vector = speaker_encoder.run(
                None, {speaker_input_name: filterbank[None].numpy()}
            )[0]
            speaker_embedding = torch.from_numpy(speaker_vector).reshape(-1).float()
        input_ids, speaker_position = build_speech_prompt(
            tokenizer, speech_config, text
        )
        payload.data = SpeechPrompt(
            input_ids=torch.tensor(input_ids, dtype=torch.long),
            speaker_position=speaker_position,
            speaker_embedding=speaker_embedding,
        )
        return payload

    return SimpleScheduler(preprocess)


def create_tts_engine_executor(
    model_path: str,
    *,
    max_seq_len: int,
    enable_acoustic_graph: bool,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    server_args_overrides: dict[str, JsonValue] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    if tp_size != 1:
        raise ValueError(f"Ming 3 TTS supports tp_size=1, got {tp_size}")
    else:
        pass
    checkpoint = resolve_checkpoint(model_path)
    speech_config = load_speech_config(Path(checkpoint))
    AutoConfig.register(Ming3TTSConfig.model_type, Ming3TTSConfig, exist_ok=True)
    ModelRegistry.models["BailingMoeV3ForCausalLM"] = Ming3TTSModel
    overrides = dict(server_args_overrides or {})
    overrides.update(
        tp_size=tp_size,
        trust_remote_code=False,
        json_model_override_args=json.dumps(
            {"architectures": ["BailingMoeV3ForCausalLM"]}
        ),
        disable_radix_cache=True,
        chunked_prefill_size=-1,
    )
    server_args = build_sglang_server_args(
        checkpoint, context_length=max_seq_len, **overrides
    )
    worker, tree_cache, request_pool, token_pool, model_config = (
        create_sglang_infrastructure(
            server_args,
            resolved_device.index or 0,
            tp_rank=tp_rank,
            nccl_port=nccl_port,
            total_gpu_memory_fraction=total_gpu_memory_fraction,
        )
    )
    tokenizer = get_tokenizer(checkpoint, trust_remote_code=False)

    def build_request(payload: StagePayload) -> Ming3TTSRequestData:
        prompt: SpeechPrompt = payload.data
        parameters = payload.request.params
        requested_steps = parameters.get("max_new_tokens")
        if requested_steps is None:
            max_decode_steps = speech_config.maximum_decode_steps
        else:
            max_decode_steps = min(requested_steps, speech_config.maximum_decode_steps)
        temperature = parameters.get("temperature")
        if temperature is None:
            temperature = speech_config.sampling_temperature
        else:
            pass
        seed = parameters.get("seed")
        noise_generator = torch.Generator(device=resolved_device)
        if seed is None:
            noise_generator.seed()
        else:
            noise_generator.manual_seed(seed)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=max_decode_steps,
            stop_token_ids=[speech_config.audio_end_token_id],
        )
        sampling_params.normalize(tokenizer)
        sampling_params.verify(model_config.vocab_size)
        request = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=prompt["input_ids"].tolist(),
            sampling_params=sampling_params,
            vocab_size=model_config.vocab_size,
            eos_token_ids={speech_config.audio_end_token_id},
        )
        request.tokenizer = tokenizer
        return Ming3TTSRequestData(
            input_ids=prompt["input_ids"],
            req=request,
            max_new_tokens=max_decode_steps,
            temperature=0.0,
            output_ids=request.output_ids,
            stage_payload=payload,
            speaker_embedding=prompt["speaker_embedding"],
            speaker_position=prompt["speaker_position"],
            sampling_temperature=float(temperature),
            noise_generator=noise_generator,
            latent_history=torch.zeros(
                (1, speech_config.latent_dim), device=resolved_device
            ),
        )

    def build_result(request: Ming3TTSRequestData) -> StagePayload:
        payload = request.stage_payload
        payload.data = {
            "latents": torch.stack(request.latents).float().cpu(),
            "usage": {
                "prompt_tokens": len(request.input_ids),
                "completion_tokens": len(request.output_ids),
            },
        }
        return payload

    stop_policy = StopPolicy(
        stop_probability_threshold=speech_config.stop_threshold,
        minimum_decode_steps=speech_config.minimum_decode_steps,
        audio_patch_token_id=speech_config.audio_patch_token_id,
        audio_end_token_id=speech_config.audio_end_token_id,
    )
    if enable_acoustic_graph:
        acoustic_graph_max_batch_size = server_args.max_running_requests
    else:
        acoustic_graph_max_batch_size = None
    model_runner = Ming3TTSModelRunner(
        worker, SGLangOutputProcessor(), stop_policy, acoustic_graph_max_batch_size
    )
    return OmniScheduler(
        tp_worker=worker,
        tree_cache=tree_cache,
        req_to_token_pool=request_pool,
        token_to_kv_pool_allocator=token_pool,
        server_args=server_args,
        model_config=model_config,
        model_runner=model_runner,
        request_builder=build_request,
        result_adapter=build_result,
        enable_async_decode=False,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler:
    resolved_device = resolve_concrete_device(device, gpu_id)
    checkpoint = Path(resolve_checkpoint(model_path))
    vae_configuration = json.loads((checkpoint / "config.json").read_text())[
        "audio_config"
    ]
    for codec_section in ("enc_kwargs", "dec_kwargs"):
        vae_configuration[codec_section]["backbone"]["_attn_implementation"] = "sdpa"
    weight_index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    vocoder_shard_names = {
        shard_name
        for weight_name, shard_name in weight_index["weight_map"].items()
        if weight_name.startswith(VOCODER_WEIGHT_PREFIX)
    }
    vocoder_weights: dict[str, torch.Tensor] = {}
    for shard_name in sorted(vocoder_shard_names):
        with safe_open(str(checkpoint / shard_name), framework="pt") as shard:
            for weight_name in shard.keys():
                if weight_name.startswith(VOCODER_WEIGHT_PREFIX):
                    vocoder_weights[weight_name.removeprefix(VOCODER_WEIGHT_PREFIX)] = (
                        shard.get_tensor(weight_name)
                    )
                else:
                    pass
    vae = AudioVAE(AudioVAEconfig(**vae_configuration)).eval()
    vae.load_state_dict(vocoder_weights)
    vae.to(device=resolved_device, dtype=torch.bfloat16)
    sample_rate = int(vae_configuration["sample_rate"])

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        latents = payload.data["latents"].to(resolved_device, dtype=torch.bfloat16)
        waveform, _, _ = vae.decode(latents.reshape(1, -1, latents.shape[-1]))
        usage = payload.data["usage"]
        payload.data = audio_waveform_payload(
            waveform.float().cpu(),
            sample_rate=sample_rate,
            modality="audio",
            source_hint="Ming-3.0-TTS",
        )
        payload.data["usage"] = usage
        return payload

    return SimpleScheduler(decode)
