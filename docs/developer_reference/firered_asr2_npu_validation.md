# FireRedASR2-LLM Ascend validation

The current PR implementation passed full-model HTTP validation on 2026-10-12
with SGLang **0.5.21** and one Ascend 910 A3 device. The earlier startup failure
on SGLang 0.5.19 was resolved by installing a matching runtime in a separate
virtual environment. No shared SGLang source or installation was changed.

The model source tested is commit `9086515e`, based on Omni main
`1ef7b8913cd2a4a3c803f1c75fafad42537356ce`. Commit `dcd8079f` corrects benchmark
WAV duration accounting; it does not change model inference.

## Environment

- ARM64, Python 3.11.15, PyTorch 2.10.0+cpu, torch-npu 2.10.0.
- CANN 9.0.0, SGLang 0.5.21, Transformers 5.12.1.
- kaldi-native-fbank 1.22.3, librosa 0.11.0, soundfile 0.13.1.
- Scoring: jiwer 4.0.0, openai-whisper 20250625, zhon 2.1.1.
- One physical Ascend 910 A3 chip with 64 GiB HBM; TP=1.
- FP32 acoustic encoder and adapter, BF16 Qwen2 decoder; greedy decoding,
  256 output tokens, graphs and torch.compile disabled.
- Merged checkpoint: `allendou/FireRedASR2-LLM-vllm`, ModelScope revision
  `25fa06878846fe089b328eca4d0c9251261c487a`. All nine safetensors shards
  (about 33.4 GB) match the mirror's SHA-256 metadata.

The NPU installation manifest on this Omni base still targets SGLang 0.5.20.
That release lacks `sglang.srt.runtime_context.SpawnRanks`. For this test,
a virtual environment inherited the prepared Ascend stack and installed the
0.5.21 ARM64 wheel with `--no-deps`, preserving the existing torch/torch-npu
pair. This validates the listed FireRed configuration, not the entire NPU
container matrix or every model registered by Omni.

## End-to-end coverage

| Check | Result |
|---|---|
| Four reference clips: serial, concurrent, SSE | 12/12 exact reference matches |
| Same checks with both caches disabled | 12/12 exact reference matches |
| JSON, text, verbose JSON | Passed |
| 48 kHz stereo upload and mono/16 kHz conversion | Exact reference match |
| FLAC upload | Exact reference match |
| 66-second upload through HTTP chunking | Exact concatenated reference; ordered segment boundaries |
| Long SSE, unsupported formats/timestamps, empty/corrupt audio, zero token limit | Expected HTTP 400/422; subsequent valid request succeeds |
| FireRed unit tests | 28 passed |
| Dataset/benchmark and TTS benchmark regression tests | 71 passed |

The extended HTTP suite has 13 cases. SSE checks require a done event,
`[DONE]`, and agreement between concatenated deltas and final text. SSE is
text streaming after a complete audio upload, not streaming audio input.

The reference clips and expected text are:

| Clip | Transcript |
|---|---|
| `hello_zh.wav` | 你好世界 |
| `hello_en.wav` | hello speech |
| `BAC009S0764W0121.wav` | 甚至出现交易几乎停滞的情况 |
| `IT0011W0001.wav` | 换一首歌 |

The first two clips are [FireRedASR2S assets](https://github.com/FireRedTeam/FireRedASR2S/tree/main/assets);
the other two are [FireRedASR examples](https://github.com/FireRedTeam/FireRedASR/tree/main/examples/wav).

## SeedTTS accuracy

Both languages use [zhaochenyang20/seed-tts-eval-arrow](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-arrow) revision
`27f4c1adee83b5b29b7c4b375f6b976324bda308`. ASR transcribes `ref_audio` and
scores against `ref_text`, rather than the separate TTS target sentence.
English uses all 1,088 rows. Chinese uses 1,088 of the original 2,020 rows:
select indices with `random.Random(20261012).sample(range(2020), 1088)` and
retain their original order. This is a fixed Chinese subset, not the full
Chinese split.

| Language | Evaluated / requested | Metric | Substitutions | Deletions | Insertions | Reference units |
|---|---:|---:|---:|---:|---:|---:|
| English | 1088 / 1088 | WER **1.7327%** | 180 | 17 | 32 | 13,216 words |
| Chinese | 1088 / 1088 | CER **0.8559%** | 78 | 10 | 65 | 17,875 characters |

These are corpus error rates, `(substitutions + deletions + insertions) /
reference units`, from non-streaming requests at concurrency 8. No requests
failed or were excluded. English uses Whisper's EnglishTextNormalizer;
Chinese removes punctuation and spaces and scores individual characters,
following the repository's shared ASR evaluator. This is not a comparison
against the official beam-search decoder.

The English cohort contains 666 distinct audio files; the Chinese cohort
contains 783. Repeated rows remain part of the requested cohorts, with audio
embedding and radix caches disabled. Their audio durations are respectively
5,105.20375 and 5,085.96367 seconds. Duration uses actual WAV frame counts;
ancillary RIFF chunks are excluded from RTF and audio-throughput calculations.
The English accuracy run's time-based ratios were recomputed from preserved
request latencies after fixing this benchmark accounting issue.

## Performance

Each measured pass uses the same 1,088 rows per language, with both caches
disabled. Concurrency levels are 1, 2, 4, and 8; concurrency 8 has three
measured passes per language. Two warmup requests per concurrency slot are
excluded from the timed window. The client and server share the host and
communicate over HTTP loopback.

| Language | Concurrency | Passes | Requests/s | Mean latency (s) | P95 latency (s) | Mean text TTFT (s) | Audio realtime multiple |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 1 | 1 | 1.466 | 0.682 | 0.874 | 0.149 | 6.878 |
| English | 2 | 1 | 2.247 | 0.890 | 1.120 | 0.229 | 10.545 |
| English | 4 | 1 | 4.147 | 0.963 | 1.237 | 0.289 | 19.460 |
| English | 8 | 3 | 5.963 | 1.339 | 1.765 | 0.338 | 27.979 |
| Chinese | 1 | 1 | 1.700 | 0.588 | 0.776 | 0.150 | 7.946 |
| Chinese | 2 | 1 | 2.598 | 0.769 | 0.996 | 0.220 | 12.146 |
| Chinese | 4 | 1 | 4.509 | 0.886 | 1.172 | 0.292 | 21.076 |
| Chinese | 8 | 3 | 6.280 | 1.272 | 1.734 | 0.364 | 29.357 |

All performance requests use SSE. TTFT starts with the HTTP POST and ends
at the first text delta, including upload, preprocessing, and server queueing.
Latency ends when the complete response is consumed. This is not incremental
audio input. Throughput divides total requests by total measured wall time;
the realtime multiple divides total audio duration by that same wall time.
Latency and TTFT statistics pool individual requests across all passes.

At concurrency 8, English throughput is 5.904, 6.041, and 5.945 requests/s
across the three passes; Chinese is 6.286, 6.291, and 6.264 requests/s.
Each row at concurrency 8 therefore aggregates 3,264 requests. The separate
non-streaming accuracy passes achieve 5.818 and 6.272 requests/s respectively.
Across accuracy and performance, **15,232 measured requests** complete with
**zero HTTP failures and zero skipped samples**. Warmup and smoke requests
are excluded from this count.

All three concurrency-8 SSE passes score 1.7327% English WER and 0.8503%
Chinese CER. Across concurrency levels, English WER ranges from 1.7100% to
1.7327%; Chinese CER ranges from 0.8448% to 0.8615%. The initial concurrency-8
JSON and SSE runs match exactly on all 1,088 English rows and 1,087 Chinese
rows. Chinese sample `00005315-00000090` differs by `他们` versus `它们`.
An isolated three-request replay agrees across serial, concurrent, and SSE
responses, including delta/final-text agreement. This does not establish
exact transcript equality across separate batching configurations or identify
the cause of the variation.

Device monitoring records 736 samples, with an observed mean interval of
5.733 seconds. The highest reported HBM utilization is 35% of 64 GiB;
NPU utilization peaks at 79%. These are coarse device-level samples, not
allocator peak-memory measurements, and omit model loading. The host is
shared and the sample interval can miss short peaks. The evaluation server
and its worker were stopped after the tests, and its port was verified closed.

## Reproduce

Starting from a prepared Ascend environment, use a separate virtual environment
for the matching runtime and evaluation packages:

```bash
python -m venv --system-site-packages .venv-firered-npu
source .venv-firered-npu/bin/activate
python -m pip install --no-deps sglang==0.5.21
python -m pip install kaldi-native-fbank==1.22.3 jiwer==4.0.0 \
  zhon==2.1.1 more-itertools==11.1.0
python -m pip install --no-deps openai-whisper==20250625
```

Select an available device and launch from the repository root, replacing the
example device and checkpoint path:

```bash
OMP_NUM_THREADS=2 ASCEND_RT_VISIBLE_DEVICES=0 \
SGLANG_OMNI_PLATFORM_SPEC=sglang_omni.platforms.npu.NPUOmniPlatform \
python -m sglang_omni.cli serve \
  --config examples/configs/firered_asr2.yaml \
  --model-path /models/firered-asr2 \
  --host 127.0.0.1 --port 8000 \
  --asr.factory.device npu \
  --asr.factory.mm_embedding_cache_size_bytes 0 \
  --asr.engine.disable_radix_cache true \
  --asr.engine.max_total_tokens 8192 \
  --asr.engine.mem_fraction_static 0.6
```

Save the four reference clips and a JSON mapping their basenames to the text
above, then run:

```bash
python examples/firered_asr2_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --audio assets/hello_zh.wav assets/hello_en.wav \
          assets/BAC009S0764W0121.wav assets/IT0011W0001.wav \
  --expected expected_transcripts.json --output native_results.json

OMP_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/unit_test/firered_asr2
```

The shared `benchmarks.tasks.asr` and `BenchmarkRunner` produce the accuracy,
latency, throughput, and SSE timing measurements. The standard
`benchmarks.eval.benchmark_asr_seedtts` CLI accepts the same local `meta.lst`
cohorts through `--meta`, with `--lang en` or `--lang zh`, `--concurrencies
1,2,4,8`, `--repeats 1`, `--stream`, and `--save-raw-dir`. Its `--warmup` flag
runs an entire discarded cohort; the recorded harness instead uses
`RunConfig(warmup=2 * concurrency)` before each measured pass.

## Independent reference and limits

Initial validation on Omni base `9c0e76ebd` compared the official Kaldi/CMVN
frontend and FP32 Conformer elementwise on all four clips: maximum absolute
error was zero. An independent all-FP32 reference used the official encoder
and adapter plus Transformers Qwen2ForCausalLM. Its greedy transcripts match
the current HTTP smoke results. Keeping the acoustic path in FP32 before
checkpoint loading matters: an initial BF16 encoder changed `hello speech`
to `哈喽 speech`.

A separate small random-weight test covered chunked prefill with 128-token
chunks and repeated-audio caching. It checked execution and cache reuse,
not recognition quality. The current full-model tests additionally cover
real HTTP long-audio chunking.

The host is shared; these measurements are not a dedicated-machine saturation
study. CUDA, tensor parallelism, multiple nodes, quantization, graph execution,
and incremental audio input have not been validated. The encoder remains
inside the language-model forward, without a separate encoder stage or graph
optimization. Exact environment versions, checkpoint hashes, and numerical
reference results are retained in the [JSON record](firered_asr2_npu_results.json).
