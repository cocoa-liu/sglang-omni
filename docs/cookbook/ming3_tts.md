# Ming-3.0-tts-tiny

[Ming-3.0-tts-tiny](https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny) synthesizes speech from text and can clone one reference voice. A Ling V3 mixture-of-experts backbone predicts one continuous 64-dimensional latent per step through a flow-matching head. A CAM++ speaker embedding conditions the prompt, and an AudioVAE decoder turns the latents into a 44.1 kHz waveform.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

Startup uses one GPU. Preprocessing runs on the CPU. The backbone and the VAE decoder share that GPU.

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-tts-tiny
```

The speech endpoint is `POST /v1/audio/speech`. The body is one complete waveform. `stream=true` still waits until that waveform is finished, then returns it as a single PCM body.

## Zero-shot speech

Leave out `ref_audio` to synthesize with a zero speaker embedding. `seed` fixes the flow-matching noise. `response_format` accepts `wav`, `mp3`, `flac`, and `pcm`. The default is `wav`.

1. Use curl

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "Get the trust fund to the bank early.",
    "response_format": "wav",
    "seed": 123
  }' \
  --output speech.wav
```

2. Use Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "model": "inclusionAI/Ming-3.0-tts-tiny",
        "input": "Get the trust fund to the bank early.",
        "response_format": "wav",
        "seed": 123,
    },
    timeout=180,
)
response.raise_for_status()
open("speech.wav", "wb").write(response.content)
```

The same request synthesizes Chinese. An empty `input` is HTTP 400.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "明天早上八点在会议室见。",
    "response_format": "wav",
    "seed": 123
  }' \
  --output speech-zh.wav
```

`temperature` overrides the checkpoint default of 1.5. `max_new_tokens` limits the latent steps and is clamped to the checkpoint maximum of 250. Speech can stop earlier, when the stop head exceeds 0.5, and not before 13 steps.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "Get the trust fund to the bank early.",
    "response_format": "mp3",
    "temperature": 0.8,
    "max_new_tokens": 40,
    "seed": 7
  }' \
  --output speech.mp3
```

## Voice cloning

`ref_audio` is one recording. A bare path is read by the server from its working directory, so start the server at the repository root when the path is `tests/data/query_to_cars.wav`. A `data:` URL is also accepted. `ref_text` is accepted and does not change the speaker embedding. A second reference is rejected.

1. Use curl

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-3.0-tts-tiny",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": "tests/data/query_to_cars.wav",
    "ref_text": "how many cars are there in the picture",
    "response_format": "wav",
    "seed": 123
  }' \
  --output cloned.wav
```

2. Use Python

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "model": "inclusionAI/Ming-3.0-tts-tiny",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "tests/data/query_to_cars.wav",
        "ref_text": "how many cars are there in the picture",
        "response_format": "wav",
        "seed": 123,
    },
    timeout=180,
)
response.raise_for_status()
open("cloned.wav", "wb").write(response.content)
```
