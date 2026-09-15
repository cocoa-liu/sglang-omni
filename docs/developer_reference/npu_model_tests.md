# NPU model regressions

This first suite covers Qwen3-TTS CustomVoice, Base and VoiceDesign via the
existing speech API. It is opt-in and has no GPU speed thresholds. Qwen3-TTS
NPU execution/configurations depend on PR #2004; do not claim this suite passes
on a main revision that does not yet contain that support.

## Run one model

In a compatible environment with exactly one reserved visible NPU, install the
source revision under test and use that model's NPU serving config:

```bash
python -m pip install pytest==8.3.5 jiwer==4.0.0 rapidfuzz==3.14.1
export ASCEND_RT_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export OMNI_RUN_NPU_TESTS=1
export OMNI_NPU_TTS_MODEL=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice
export OMNI_NPU_TTS_CONFIG=/path/to/qwen3_tts_0_6b_customvoice_npu.yaml
export OMNI_NPU_TTS_TASK=CustomVoice
export OMNI_NPU_TTS_OUTPUT=/tmp/qwen3-tts-regression
python -m pytest tests/test_model/test_npu_tts.py -v -s \
  --junitxml=/tmp/qwen3-tts-regression.xml
```

`jiwer` / `rapidfuzz` are test-only dependencies required by the existing
`tests.test_model` shared plugin, even when this suite does not run WER. They
need not be added to the inference image's runtime dependency list.

The fixture starts and stops its own server on an available loopback port. Set
`OMNI_NPU_TTS_PORT` to reserve a specific port; an occupied port is an error, not
a server to reuse. Automatic port selection avoids reusing a previous run's
connections while they are still in TIME_WAIT.
Local model and config paths are mandatory. When disabled, tests skip; when
enabled, missing models, unsupported hardware and startup failures fail the run.

For Base, select the corresponding model/config and set:

```bash
export OMNI_NPU_TTS_TASK=Base
export OMNI_NPU_TTS_REFERENCE=/references/voice.wav
export OMNI_NPU_TTS_REFERENCE_TEXT='The exact words spoken in this reference.'
```

The reference must be available to the server. For VoiceDesign, select its
model/config and set `OMNI_NPU_TTS_TASK=VoiceDesign`; no reference is needed.
Run variants sequentially on a reserved card, not concurrently on the same card.

## Coverage and evidence

- English/Chinese non-streaming speech and a fully consumed PCM stream.
- Two concurrent requests (queueing is allowed; this does not prove batching).
- A valid request after rejecting a malformed request.
- Status, sample format, completeness, non-silent output and broad short-prompt
  duration bounds. Durations/first bytes are recorded, not compared to GPU limits.
- Server log, request metadata, WAV/PCM outputs and optional pytest JUnit XML.

This is a functional smoke/regression suite, not a quality benchmark: passing
does not establish intelligibility, speaker similarity or absence of reference
text leakage. WER/speaker-sim evaluation and model-specific performance baselines
must be added with validated assets and thresholds. Network chunk boundaries
are not assumed to equal model chunk boundaries.

No global CUDA/NPU process cleanup is used. The shared `managed_omni_server`
context stops only the server it started. Prefer a disposable container for CI
so cancellation can also clean up descendants. Future runner integration should
record both the source SHA and image digest, archive logs on failure and avoid
automatic retries of deterministic failures.
