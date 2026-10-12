# FireRedASR2-LLM

FireRedASR2-LLM runs as one native ASR stage: an 80-bin Kaldi fbank frontend,
CMVN normalization, the FireRed Conformer encoder, a frame-concatenating
adapter, and SGLang's Qwen2 decoder. Omni registers the model locally; no
FireRed implementation or source patch in upstream SGLang is required.
SGLang still provides the decoder, attention backend, KV cache, and sampling.

## Checkpoint and installation

Follow the appropriate [installation guide](../get_started/installation.md)
([Ascend NPU](../get_started/installation_npu.md)), then install the frontend:

```bash
pip install 'kaldi-native-fbank>=1.20.0'
hf download allendou/FireRedASR2-LLM-vllm --local-dir /models/firered-asr2
```

The `firered-asr2` installation extra also supplies this dependency. Use the
merged HF checkpoint linked by the
[official serving guide](https://github.com/FireRedTeam/FireRedASR2S).
It contains `audio_encoder_conf`, the tokenizer, CMVN statistics in
`preprocessor_config.json`, and safetensors with `encoder.*`,
`encoder_projector.*`, and `llm.*` weights. Its name contains `vllm`, but this
integration does not depend on vLLM. The original `model.pth.tar` plus separate
Qwen2 directory is not an input format for this loader.

## Serve

```bash
sgl-omni serve \
  --config examples/configs/firered_asr2.yaml \
  --model-path /models/firered-asr2 \
  --host 127.0.0.1 --port 8000
```

For Ascend, set `ASCEND_RT_VISIBLE_DEVICES` to the selected physical card and
add `--asr.factory.device npu`. The stage uses logical device 0 inside that
visibility set. The example uses a `bfloat16` Qwen2 decoder, eight concurrent
requests, and `mem_fraction_static=0.70`. The Conformer and adapter always
retain `float32` weights and compute; projected embeddings are converted to
the decoder dtype. Casting the acoustic path to BF16 changed an English
transcript in the validation set. Lower concurrency or the memory fraction to leave
more encoder activation headroom. The encoder processes at most four audio
items per forward. Decode and prefill graphs are disabled for this initial
implementation.

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json

# Stream generated text through SSE:
curl -N http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=json -F stream=true
```

The endpoint supports `json`, `text`, and `verbose_json`. Streaming accepts
`json` or `text`; use `transcript.text.done` as the final transcript.
Audio is converted to mono 16 kHz. The transcription prompt is fixed to
`请转写音频为文字`; `language` and `prompt` do not alter model conditioning.
The default decode is greedy, with up to 256 new tokens. There is no beam
search in this adapter, so results need not equal the official beam-search
defaults. Translation, word timestamps, and speaker diarization are not
provided by this model adapter.

Non-streaming uploads longer than 30 seconds use the shared audio chunking
path. Each direct model request is limited to 40 seconds. SSE streams the
text generated from a complete uploaded clip; it is not incremental audio
input.

## Reproduce smoke checks

With a server running and speech clips available locally:

```bash
python examples/firered_asr2_smoke.py \
  --base-url http://127.0.0.1:8000 \
  --audio /path/to/chinese.wav /path/to/english.wav \
  --output firered-results.json

pytest -q tests/unit_test/firered_asr2
```

The smoke script compares serial, concurrent, and SSE results. Pass
`--expected reference.json` to also check exact reference transcripts; the
file is a JSON object mapping audio basenames to text. Unit tests cover
frontend normalization, audio token counts, variable-length batches, weight
loading, and device placement. These checks are not a corpus WER evaluation
or a throughput benchmark.

See the [Ascend validation record](../developer_reference/firered_asr2_npu_validation.md)
for the tested environment, reference outputs, launch command, and current
PR revalidation status. Full-model results were collected on the earlier
development base; revalidation against the current Omni main requires a
matching SGLang runtime.
