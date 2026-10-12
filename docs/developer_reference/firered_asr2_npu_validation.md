# FireRedASR2-LLM Ascend validation

Initial end-to-end validation ran on 2026-10-12 using one Ascend 910 A3
device, in the development tree based on Omni commit `9c0e76ebd`. The implementation
registers its architecture in Omni and uses SGLang's Qwen2 decoder, attention,
KV cache, and sampling. No FireRed-specific patch to SGLang is required.

## PR branch verification status

The PR branch is based on Omni main `1ef7b8913cd2a4a3c803f1c75fafad42537356ce`.
Its 28 unit tests pass in the same NPU environment (12.75 seconds), and all
pre-commit checks pass except the unrelated Rust formatting hook, which is
skipped because Cargo is unavailable locally.

**End-to-end revalidation on this newer Omni base is blocked.** The installed
SGLang `0.5.19.dev852+gf58665451` lacks `SpawnRanks` from
`sglang.srt.runtime_context`, now imported by Omni's shared `ModelWorker`.
Startup fails before FireRed model loading. Explicitly selecting
`SGLANG_OMNI_PLATFORM_SPEC=sglang_omni.platforms.npu.NPUOmniPlatform` resolves
an earlier platform-detection issue, but does not resolve this missing API.

The full-model and numerical-parity results below describe the initial
implementation on the earlier development base. They do **not** establish
end-to-end compatibility of the current PR branch. Before making the PR ready
for merge, repeat the HTTP smoke checks with a SGLang runtime matching the
current Omni installation requirements. No shared SGLang source was patched
or upgraded for this attempt.

## Environment and checkpoint

- Python 3.11.15, PyTorch 2.10.0+cpu, torch-npu 2.10.0.
- SGLang `0.5.19.dev852+gf58665451`, commit
  `f5866545186f27a8a01c1a083e8c32809b5506f3`.
- Transformers 5.12.1, kaldi-native-fbank 1.22.3.
- The independent reference also uses kaldiio 2.18.1 and accelerate 1.15.0.
- Merged checkpoint: `allendou/FireRedASR2-LLM-vllm`, downloaded from ModelScope
  revision `25fa06878846fe089b328eca4d0c9251261c487a`.
  All nine safetensors shards (about 33.4 GB) match the SHA-256 digests in the
  mirror metadata.

The exact versions, checkpoint digests, source digests, and transcripts are
included in the [JSON results](firered_asr2_npu_results.json).

## Precision and reference

The acoustic encoder and adapter keep FP32 weights and compute. They are
converted to FP32 **before loading** the checkpoint so that acoustic weights
never pass through BF16 rounding. The saved relative positional encoding
buffer is required as well. Only the projected embeddings are cast to the
Qwen2 decoder's BF16 dtype.

An initial all-BF16 run transcribed `hello speech` as `哈喽 speech`. Retaining
FP32 for the acoustic path removed this difference; a unit test covers the
precision policy.

The independent full reference uses the official FireRedASR2 Conformer,
adapter, Kaldi/CMVN frontend, and Transformers Qwen2ForCausalLM, all in FP32.
Both implementations use the same prompt, greedy decoding, a 256-token
output limit, EOS, and text trimming. This comparison does not use the
official beam-search default.

## Results

| Audio | Reference transcript | Omni serial / concurrent / SSE |
|---|---|---|
| `hello_zh.wav` | 你好世界 | Exact match in all three modes |
| `hello_en.wav` | hello speech | Exact match in all three modes |
| `BAC009S0764W0121.wav` | 甚至出现交易几乎停滞的情况 | Exact match in all three modes |
| `IT0011W0001.wav` | 换一首歌 | Exact match in all three modes |

The first two clips come from [FireRedASR2S assets](https://github.com/FireRedTeam/FireRedASR2S/tree/main/assets).
The other two come from [FireRedASR examples](https://github.com/FireRedTeam/FireRedASR/tree/main/examples/wav).

- Real-checkpoint frontend features and FP32 Conformer outputs match the
  official implementation elementwise: maximum absolute error 0 on all four clips.
- Twelve full-model HTTP requests pass: four serial, four submitted
  concurrently, and four SSE. Concatenated SSE deltas, done text,
  non-streaming text, and reference transcripts agree.
- Twenty-eight unit tests cover frontend normalization, token lengths,
  variable-length batches, request validation, weight completeness, model
  discovery, device placement, and mixed precision.
- A separate small random-weight model checks 128-token chunked prefill and
  repeated-audio caching. Two approximately 23-second clips are each requested
  twice; the encoder runs twice in total and repeated outputs agree. This
  checks execution and caching, not transcription quality.

The full-model run uses `max_total_tokens=4096`, `mem_fraction_static=0.6`,
and disabled graphs. The load log reports about 17.9 GiB of additional device
memory during model loading; this is neither peak usage nor a minimum memory
requirement. Recorded request times are smoke-test observations, not benchmarks.

## Reproduce the HTTP checks

After installing the [model dependencies](../cookbook/firered_asr2.md), select
an available device, download the checkpoint, and start the server from the
repository root. Replace the example device and model path for your deployment.

```bash
OMP_NUM_THREADS=2 ASCEND_RT_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/firered_asr2.yaml \
  --model-path /models/firered-asr2 \
  --host 127.0.0.1 --port 8000 \
  --asr.factory.device npu \
  --asr.engine.max_total_tokens 4096 \
  --asr.engine.mem_fraction_static 0.6
```

Save the four clips under `assets/` and a JSON object mapping their basenames
to the reference transcripts above as `expected_transcripts.json`, then run:

```bash
python examples/firered_asr2_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --audio assets/hello_zh.wav assets/hello_en.wav \
          assets/BAC009S0764W0121.wav assets/IT0011W0001.wav \
  --expected expected_transcripts.json --output native_results.json

OMP_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/unit_test/firered_asr2
```

## Scope

This is single-device functional and sample-consistency validation. It does
not establish corpus WER/CER or throughput. CUDA, tensor parallelism,
multi-node execution, quantization, and graph modes have not been validated.
Real-checkpoint HTTP chunking for long recordings has not been evaluated
separately. The encoder runs inside the language-model forward; there is no
separate pre-LM encoder service or encoder graph optimization in this version.
