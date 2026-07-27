# SGLang-Omni 昇腾 A3 人工验证手册

**更新日期**: 2026-07-22

**环境**: A3 `113.46.38.25` 或 `113.46.46.40`，Docker 容器 `lc-l3-test`

**模型**: `/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct`

本文用于人工验证 Qwen3-Omni 的文字、多模态理解和语音输出。本文中的 NPU Talker 默认无 seed 改动可绕过 SGLang seeded sampler 的 `murmur_hash32_kernel` 编译失败，但截至 2026-07-22 的人工试听显示其生成 WAV 语义混乱，**不能作为可用 TTS 修复交付**。下文的 Speech 章节仅用于检查链路和定位问题；必须完成第 5.5 节的听感或独立 ASR 验证后，才能判定语音功能通过。

相关实测记录：

- `2026-07-22-qwen3-omni-tts-remote-40-investigation.md`
- `2026-07-22-qwen3-omni-multimodal-to-speech-e2e.md`

## 1. 连接和资源检查

本机 SSH 配置损坏时，使用 `-F /dev/null` 绕过系统配置：

```bash
ssh -F /dev/null -i ~/.ssh/id_ed25519 -p 22 \
  -o BatchMode=yes -o StrictHostKeyChecking=no root@<IP>
```

检查容器、服务和 NPU：

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep '^lc-l3-test'
docker exec lc-l3-test npu-smi info
docker exec lc-l3-test pgrep -af 'sgl-omni serve'
```

启动 Speech 服务前，NPU 0-8 应没有不属于本次验证的进程。不要直接执行文档外的 `pkill -9 -f sglang` 或重启容器；先通过 `npu-smi info` 和 `ps -fp <PID>` 确认进程归属。若刚停止过本服务，需确认其 worker 已退出，否则会因残留显存触发 `Not enough memory`。

## 2. 确认 NPU TTS 修复已在代码中

本修复的目标是让 NPU 上未显式指定 `seed` 的 TTS 请求不进入 SGLang 的 seeded sampler。检查以下两点：

```bash
docker exec lc-l3-test sed -n '1058,1072p' \
  /home/l00951280/sglang-omni/sglang_omni/models/qwen3_omni/request_builders.py

docker exec lc-l3-test sed -n '925,945p' \
  /home/l00951280/sglang-omni/sglang_omni/models/qwen3_omni/components/talker.py

docker exec lc-l3-test sed -n '1175,1192p' \
  /home/l00951280/sglang-omni/sglang_omni/models/qwen3_omni/components/talker.py
```

预期：

- `request_builders.py` 不再为默认 Talker 请求按 request id 自动写入 `seed`。
- `talker.py` 在 `self._device.type == "npu"` 时将 `SamplingBatchInfo.sampling_seed` 设为 `None`。
- NPU 上显式传入 `seed` 暂不支持，应得到清晰的错误；不要把 seed 静默丢弃。

## 3. 启动 Speech 服务

以下命令占用 NPU 0-8：Thinker TP=8 使用 0-7，Talker 使用 8。日志独立写入 `/tmp/sgl-omni-run-npu-unseeded.log`。

```bash
docker exec -d lc-l3-test bash -c '
export ASCEND_AUTO_CONNECT=0
export HCCL_NPU_SOCKET_PORT_RANGE=12000-22000

exec sgl-omni serve \
  --model-path /home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 \
  --talker-gpu 8 \
  --mem-fraction-static 0.75 \
  > /tmp/sgl-omni-run-npu-unseeded.log 2>&1
'
```

等待就绪：

```bash
docker exec lc-l3-test grep 'Application startup complete' \
  /tmp/sgl-omni-run-npu-unseeded.log
docker exec lc-l3-test curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8000/health
```

预期分别出现 `Application startup complete` 和 `200`。

## 4. 基础文字到语音链路验证

```bash
docker exec lc-l3-test curl -sS -D /tmp/tts-headers.txt \
  -o /tmp/tts-response.wav \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","input":"你好世界","voice":"default"}' \
  http://127.0.0.1:8000/v1/audio/speech

docker exec lc-l3-test od -An -t x1 -N 12 /tmp/tts-response.wav
```

链路验收标准：

- HTTP 状态为 `200`，`content-type` 为 `audio/wav`。
- 文件头为 `52 49 46 46 ... 57 41 56 45`，即 `RIFF...WAVE`。
- 日志中没有 `murmur_hash32_kernel`、`MLIRCompilationError`、`Failed to compile BiShengLIR`。

这些检查不代表语音内容正确；仍需执行第 5.5 节。

## 5. 多模态输入到文字和语音链路验证

使用 `/v1/chat/completions`，在请求中指定 `modalities=["text", "audio"]`。响应是 JSON：`choices[0].message.content` 为文字，`choices[0].message.audio.data` 为 base64 编码的 WAV。

测试媒体已在容器中：

```bash
docker exec lc-l3-test ls -lh \
  /home/l00951280/sglang-omni/tests/data/cars.jpg \
  /home/l00951280/sglang-omni/tests/data/query_to_cars.wav \
  /home/l00951280/sglang-omni/tests/data/draw.mp4
```

### 5.1 图像到语音

```bash
docker exec lc-l3-test curl -sS -o /tmp/mm-image.json \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"请用一句中文描述图片中的主要物体。"}],
    "images":["/home/l00951280/sglang-omni/tests/data/cars.jpg"],
    "modalities":["text","audio"],
    "audio":{"voice":"default","format":"wav"},
    "max_tokens":48,
    "stream":false
  }' http://127.0.0.1:8000/v1/chat/completions
```

预期：HTTP 200，文字描述汽车，且响应包含 `message.audio.data`。

### 5.2 音频到语音

```bash
docker exec lc-l3-test curl -sS -o /tmp/mm-audio.json \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"请用一句中文说明这段音频中讨论的内容。"}],
    "audios":["/home/l00951280/sglang-omni/tests/data/query_to_cars.wav"],
    "modalities":["text","audio"],
    "audio":{"voice":"default","format":"wav"},
    "max_tokens":48,
    "stream":false
  }' http://127.0.0.1:8000/v1/chat/completions
```

预期：HTTP 200，文字应表达“询问图片中有多少辆汽车”这一语义，且响应包含音频。

### 5.3 视频到语音

长视频会显著增加视觉 token。人工验收先使用两帧：

```bash
docker exec lc-l3-test curl -sS -o /tmp/mm-video.json \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"请用一句中文说明视频中人物手里拿着什么。"}],
    "videos":["/home/l00951280/sglang-omni/tests/data/draw.mp4"],
    "video_max_frames":2,
    "modalities":["text","audio"],
    "audio":{"voice":"default","format":"wav"},
    "max_tokens":48,
    "stream":false
  }' http://127.0.0.1:8000/v1/chat/completions
```

预期：HTTP 200，文字应表达“白色触控笔”这一语义，且响应包含音频。

该命令使用 `-o /tmp/mm-video.json`，因此响应保存在 **容器内** 的该路径，不会输出到终端。查看文字并将 base64 音频提取成可播放的 WAV：

```bash
docker exec lc-l3-test python3 -c '
import base64, json
d = json.load(open("/tmp/mm-video.json"))
m = d["choices"][0]["message"]
print("text:", m.get("content"))
open("/tmp/mm-video.wav", "wb").write(base64.b64decode(m["audio"]["data"]))
'

docker exec lc-l3-test ls -lh /tmp/mm-video.json /tmp/mm-video.wav
docker cp lc-l3-test:/tmp/mm-video.wav ./mm-video.wav
```

提取后的音频在容器内 `/tmp/mm-video.wav`，最后一条命令将其复制到执行 `docker cp` 的宿主机当前目录。

### 5.4 统一检查多模态响应中的 WAV

以下命令适用于上述任一 JSON 响应：

```bash
docker exec lc-l3-test python3 -c '
import base64, json, sys
d = json.load(open(sys.argv[1]))
m = d["choices"][0]["message"]
raw = base64.b64decode(m["audio"]["data"])
print("text:", m.get("content"))
print("audio_bytes:", len(raw))
print("audio_magic:", raw[:12].hex())
' /tmp/mm-image.json
```

验收标准：`audio_bytes > 44`，且 `audio_magic` 以 `52494646` 开头、在前 12 字节中包含 `57415645`；这分别是 RIFF 和 WAVE 标识。

### 5.5 人工听感和语义验收（必做）

WAV 文件有效不等于语音内容正确。将 JSON 内的音频导出后，人工试听并核对它是否说出了文字回答：

```bash
docker exec lc-l3-test python3 -c '
import base64, json
d = json.load(open("/tmp/mm-video.json"))
m = d["choices"][0]["message"]
print("expected text:", m.get("content"))
open("/tmp/mm-video.wav", "wb").write(base64.b64decode(m["audio"]["data"]))
'
docker cp lc-l3-test:/tmp/mm-video.wav ./mm-video.wav
```

本视频 fixture 的文字预期为“一个白色的触控笔”。试听 WAV 时应能听到等价内容；如果是混乱、无意义或与文字不一致的语音，则 **Speech 验证失败**，即使 HTTP、WAV 文件头和日志均正常。

当前已知状态：默认无 seed 规避在本环境能返回 WAV，但人工试听为混乱语音。不要将其标记为 TTS 通过。

## 6. 日志和故障定位

```bash
docker exec lc-l3-test grep -n \
  'POST /v1/chat/completions\|POST /v1/audio/speech' \
  /tmp/sgl-omni-run-npu-unseeded.log | tail -n 20

docker exec lc-l3-test grep -n \
  'murmur_hash32_kernel\|MLIRCompilationError\|Failed to compile BiShengLIR' \
  /tmp/sgl-omni-run-npu-unseeded.log
```

| 现象 | 根因与处理 |
|---|---|
| 默认 TTS 返回 500，日志含 `murmur_hash32_kernel` | NPU Talker 仍在使用 seeded PyTorch sampler。该路径受当前 Triton/BiSheng 编译故障影响。 |
| HTTP 200 且 WAV 有效，但听感混乱 | 当前无 seed 随机采样规避仅恢复了链路，未恢复可用语音。下一步应在同一输入上对比 greedy `argmax` 与随机采样；不能以文件格式替代语义验收。 |
| 进程启动时报 `Not enough memory` | 上一服务的 worker 可能未退出。先根据 `npu-smi info` 的 PID 和 `ps -fp <PID>` 确认归属，只停止本次验证遗留的进程；确认 NPU 0-8 显存回收后重启。 |
| TP=8 初始化卡住 | 确认 `ASCEND_AUTO_CONNECT=0` 已导出，并避免 HCCL 端口冲突。 |
| 视频请求异常或过慢 | 降低 `video_max_frames`，先从 2 帧开始；不要直接用完整长视频作为冒烟验证。 |
| NPU 显式 seed 请求失败 | 当前运行时不支持 NPU seeded sampling。这是已知限制，应返回明确参数错误；升级 CANN/torch_npu/Ascend Triton 后再验证恢复确定性采样。 |

## 7. 停止与回滚

仅在确认服务 PID 属于本次验证时停止它：

```bash
docker exec lc-l3-test pgrep -af 'sgl-omni serve'
docker exec lc-l3-test kill -TERM <服务PID>
```

若需要回滚 NPU 无 seed 修复，恢复部署前备份的 `request_builders.py` 和 `talker.py`，然后重启服务。不要使用广泛匹配的强制杀进程命令，以免中断其他任务。

## 8. 启动参数速查表

| 参数 | Text-only | Speech |
|---|---|---|
| 模式 | `--text-only` | 默认 Speech |
| NPU 数 | 8 | 9 |
| Thinker | `--thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7` | 同左 |
| Talker | 不需要 | `--talker-gpu 8` |
| `mem-fraction-static` | `0.75` | `0.75` |
| `ASCEND_AUTO_CONNECT` | `0` | `0` |
| `HCCL_NPU_SOCKET_PORT_RANGE` | `12000-22000` | `12000-22000` |

## Residual sampler regression check

When testing an Ascend workaround, do not accept HTTP 200 or a valid WAV header
as success. After obtaining /home/cocoa/lc/qwen3_omni_tts_residual_sampler.wav,
listen to it and verify it intelligibly says 你好世界. The residual codec groups
must use the reference sampling contract (top_k=50, top_p=0.8, stochastic
sampling); replacing those groups with argmax can return a valid but unintelligible
waveform.

### Follow-up listening result and environment reset

Manual listening of `/home/cocoa/lc/qwen3_omni_tts_residual_sampler.wav` found it
was still unintelligible rather than `你好世界`. Therefore matching only the residual
codec sampler is necessary but insufficient; do not mark this attempt as a TTS fix.

The prior launcher shutdown left orphan multiprocessing workers that retained all
NPU memory. The approved recovery is `docker restart lc-l3-test`, which clears the
test container's complete process tree before the next clean service startup.

## 2026-07-22: Talker feedback hidden-state correction

### Root-cause candidate

The reference Transformers implementation builds the next Talker input from the
layer-0 codec embedding, intermediate code-predictor hidden states, and only the
final residual codec embedding. The SGLang-Omni implementation instead summed the
embedding of every residual codec group. That changes the autoregressive Talker
input after the first frame and is consistent with valid WAV output that contains
unintelligible speech.

### Fix

For residual groups 1 through N-2, run the predictor token and add its returned
hidden state to `_output_embeds`; add only the last residual group's embedding.
Backup: `talker.py.bak-20260722-before-feedback-hidden`.

After `docker restart lc-l3-test`, the patched service started cleanly on port
8000. `POST /v1/audio/speech` for `你好世界` returned HTTP 200 and produced:

- local file: `/home/cocoa/lc/qwen3_omni_tts_feedback_hidden.wav`
- format: 24 kHz mono PCM WAV
- bytes: 279164
- SHA-256: `b64a67b603238774835c90c4a4f601eed386200ebfb458c7013c477a8af53c47`

Status remains pending human listening. The only pass criterion is intelligible
speech of `你好世界`.

## 2026-07-22: full-sequence Code2Wav isolation

The official Code2Wav applies a time-axis Transformer to all codes. The prior
non-streaming scheduler decoded 10-code windows with only 25 codes of left context
and concatenated their waveforms; this is not guaranteed equivalent. For this
isolation test, non-streaming requests accumulate all codec frames and run Code2Wav
once at request completion. Streaming behavior is unchanged.

Backup: `code2wav_scheduler.py.bak-20260722-before-full-sequence-nonstreaming`.
After container restart, the service started on port 8000. `你好世界` returned HTTP
200 with `/home/cocoa/lc/qwen3_omni_tts_full_code2wav.wav`, a 24 kHz mono PCM WAV
of 286934 bytes (SHA-256
`4d2ad49ce84a9a8a95e26c7c7f95e2de5af3bc841c37a545ae53045cc0a942e7`).

Status: pending manual listening. If still unintelligible, codec code generation
rather than WAV framing or incremental Code2Wav assembly is the active root cause.
