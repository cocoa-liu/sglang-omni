# SPDX-License-Identifier: Apache-2.0
"""Check serial, concurrent, and streamed transcription against one server.

Run from the repository root:
    python examples/firered_asr2_smoke.py --audio /path/to/zh.wav /path/to/en.wav
An optional --expected JSON object maps audio basenames to reference transcripts.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def transcribe(base_url: str, path: Path, *, stream: bool = False):
    start = time.perf_counter()
    with (
        httpx.Client(timeout=300) as client,
        path.open("rb") as audio,
        client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/audio/transcriptions",
            files={"file": (path.name, audio, "audio/wav")},
            data={"response_format": "json", "stream": str(stream).lower()},
        ) as response,
    ):
        response.raise_for_status()
        if stream:
            deltas, done, ended = [], None, False
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = line[6:]
                if event == "[DONE]":
                    ended = True
                    break
                event = json.loads(event)
                if event["type"] == "transcript.text.delta":
                    deltas.append(event["delta"])
                elif event["type"] == "transcript.text.done":
                    done = event["text"]
                elif event["type"] == "error":
                    raise RuntimeError(event)
            assert ended and done is not None, "Incomplete SSE response"
            assert "".join(deltas).strip() == done, "SSE deltas differ from done"
            text = done
        else:
            response.read()
            text = response.json()["text"]
    assert text.strip(), f"Empty transcript: {path.name}"
    return {
        "file": path.name,
        "text": text,
        "elapsed_s": round(time.perf_counter() - start, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--audio", nargs="+", type=Path, required=True)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path, default=Path("firered-results.json"))
    args = parser.parse_args()
    serial = [transcribe(args.base_url, path) for path in args.audio]
    with ThreadPoolExecutor(max_workers=len(args.audio)) as pool:
        concurrent = list(
            pool.map(lambda path: transcribe(args.base_url, path), args.audio)
        )
    streamed = [transcribe(args.base_url, path, stream=True) for path in args.audio]
    expected = json.loads(args.expected.read_text()) if args.expected else None
    report = {"serial": serial, "concurrent": concurrent, "streamed": streamed}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for single, batch, stream in zip(serial, concurrent, streamed, strict=True):
        assert single["text"] == batch["text"] == stream["text"], single["file"]
        if expected is not None:
            assert single["text"] == expected[single["file"]], single
        print(f"{single['file']}: {single['text']}")
    print(f"Passed serial/concurrent/SSE checks; results: {args.output}")


if __name__ == "__main__":
    main()
