# Qwen3-Omni 昇腾 NPU 复现与操作手册

## 1. 文档定位

本文供接收方在 Ascend A3 环境复现本次阶段性交付结果。按照本文操作，应得到：

- Qwen3-Omni 服务成功启动；
- 文本、图片、音频、视频输入能够得到正确或语义相符的文字结果；
- TTS 和多模态语音响应能够返回格式合法的 WAV；
- 当前 WAV 人工试听仍为乱码，因此语音质量验收失败。

本文不要求读者预先了解 SGLang-Omni，但执行者需要具备 Linux、Docker、SSH 和 HTTP 请求的基本经验。

## 2. 先理解三层执行环境

本次环境包含三层，路径和命令不能混用：

| 层级 | 示例 | 作用 |
| --- | --- | --- |
| 当前工作机 | `/home/cocoa/lc/` | 保存文档、脚本和最终拉回的 WAV。 |
| 远端宿主机 | `root@113.46.46.40` | 运行 Docker，执行 `docker exec`、`docker cp`。 |
| 容器 | `lc-l3-test` | 运行 SGLang-Omni、模型和 HTTP 服务；容器内 `/tmp` 与宿主机 `/tmp` 不是同一目录。 |

下文若无特别说明，命令均在远端宿主机执行。

从当前工作机连接远端宿主机：

```bash
ssh -F /dev/null -i /home/cocoa/.ssh/id_ed25519 -p 22 \
  -o BatchMode=yes -o StrictHostKeyChecking=no root@113.46.46.40
```

## 3. 环境基线

### 3.1 固定路径和资源

| 项目 | 值 |
| --- | --- |
| 容器 | `lc-l3-test` |
| 源码 | `/home/l00951280/sglang-omni` |
| 模型 | `/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct` |
| 服务端口 | `8000` |
| Thinker | TP=8，NPU 0--7 |
| Talker | NPU 8 |
| 日志 | 容器内 `/tmp/sgl-omni-run-npu.log` |

当前资源配置只用于复现，不是生产推荐值。`mem_fraction_static=0.75` 是验证环境中能够启动的经验参数，显存占用仍待治理。

### 3.2 记录软件和代码版本

开始测试前保存以下输出，后续结果才能追溯：

```bash
docker exec lc-l3-test bash -lc '
python3 -V
python3 -c "import torch, torch_npu, transformers; print(\"torch\", torch.__version__); print(\"torch_npu\", torch_npu.__version__); print(\"transformers\", transformers.__version__)"
pip show sglang sglang-omni | grep -E "^(Name|Version):"
git -C /home/l00951280/sglang-omni rev-parse HEAD
git -C /home/l00951280/sglang-omni status --short
'

docker exec lc-l3-test npu-smi info
```

如果工作树存在未提交修改，应把 `git status --short` 一并归档；只记录 commit 不能完整表示测试代码。

## 4. 启动前检查

### 4.1 检查容器、端口和残留进程

```bash
docker ps --format '{{.Names}}\t{{.Status}}' | grep '^lc-l3-test'
docker exec lc-l3-test npu-smi info
docker exec lc-l3-test pgrep -af 'sgl-omni serve'
docker exec lc-l3-test bash -lc 'ss -ltnp | grep ":8000 " || true'
```

确认 NPU 0--8 上没有无关进程。不要直接执行宽范围 `pkill -9`；先使用 `npu-smi info` 的 PID 和 `ps -fp <PID>` 判断归属。

若确认容器只承载本次验证，且上一轮服务留下 worker 或显存，可执行：

```bash
docker restart lc-l3-test
```

重启后再次检查 NPU 0--8 是否回到空闲基线。若显存没有回收，不要继续启动新服务。

### 4.2 检查模型和测试数据

```bash
docker exec lc-l3-test test -r \
  /home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct/config.json

docker exec lc-l3-test ls -lh \
  /home/l00951280/sglang-omni/tests/data/cars.jpg \
  /home/l00951280/sglang-omni/tests/data/query_to_cars.wav \
  /home/l00951280/sglang-omni/tests/data/draw.mp4
```

### 4.3 检查 TTS 阶段性规避

不要依赖固定行号，使用关键符号检查：

```bash
docker exec lc-l3-test grep -nE \
  'sampling_seed|explicit.*seed|multinomial|top_k|top_p' \
  /home/l00951280/sglang-omni/sglang_omni/models/qwen3_omni/request_builders.py \
  /home/l00951280/sglang-omni/sglang_omni/models/qwen3_omni/components/talker.py
```

检查目标：

- 默认 NPU TTS 请求不自动注入 seed；
- NPU Talker 使用原生 top-k/top-p/temperature/`multinomial`；
- 显式 seed 不被静默忽略。

该检查只确认规避路径存在，不代表语音质量通过。

## 5. 启动服务

```bash
docker exec -d lc-l3-test bash -lc '
export ASCEND_AUTO_CONNECT=0
export HCCL_NPU_SOCKET_PORT_RANGE=12000-22000
exec sgl-omni serve \
  --model-path /home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 \
  --talker-gpu 8 \
  --mem-fraction-static 0.75 \
  > /tmp/sgl-omni-run-npu.log 2>&1
'
```

模型加载需要时间。重复执行以下检查，直至就绪或日志出现明确错误：

```bash
docker exec lc-l3-test grep 'Application startup complete' /tmp/sgl-omni-run-npu.log
docker exec lc-l3-test curl -sS -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:8000/health
```

就绪标准：

- 日志出现 `Application startup complete`；
- `/health` 返回 `200`；
- `npu-smi info` 中的进程和目标卡位与启动配置一致；
- 日志中没有 OOM、stage worker 退出或 HCCL 初始化失败。

失败时查看：

```bash
docker exec lc-l3-test tail -n 250 /tmp/sgl-omni-run-npu.log
```

## 6. 按顺序执行验收用例

建议按“文字 → 图片 → 音频 → 视频 → TTS”的顺序执行。前一项失败时先定位，不要用后续结果掩盖基础链路问题。

### 6.1 纯文字理解

```bash
docker exec lc-l3-test curl -sS \
  -o /tmp/text-response.json \
  -w 'HTTP %{http_code}\n' \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"default",
    "messages":[{"role":"user","content":"请只回答：你好世界"}],
    "modalities":["text"],
    "max_tokens":16,
    "stream":false
  }' http://127.0.0.1:8000/v1/chat/completions

docker exec lc-l3-test python3 -c '
import json
d = json.load(open("/tmp/text-response.json"))
print(d["choices"][0]["message"].get("content"))
'
```

预期：HTTP 200，输出包含“你好世界”，不包含服务错误 JSON。

### 6.2 图像理解

```bash
docker exec lc-l3-test curl -sS \
  -o /tmp/mm-image.json \
  -w 'HTTP %{http_code}\n' \
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

预期：HTTP 200，文字描述多辆汽车；响应包含 `message.audio.data`。

### 6.3 音频理解

```bash
docker exec lc-l3-test curl -sS \
  -o /tmp/mm-audio.json \
  -w 'HTTP %{http_code}\n' \
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

预期：HTTP 200，文字表达“询问图片中有多少辆汽车”；响应包含音频字段。

### 6.4 视频理解

```bash
docker exec lc-l3-test curl -sS \
  -o /tmp/mm-video.json \
  -w 'HTTP %{http_code}\n' \
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

预期：HTTP 200，文字表达“一个白色的触控笔”；响应包含音频字段。

### 6.5 基础 TTS

```bash
docker exec lc-l3-test curl -sS \
  -D /tmp/tts-headers.txt \
  -o /tmp/tts-nihao.wav \
  -w 'HTTP %{http_code}\n' \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","input":"你好世界","voice":"default"}' \
  http://127.0.0.1:8000/v1/audio/speech

docker exec lc-l3-test sed -n '1,20p' /tmp/tts-headers.txt
docker exec lc-l3-test od -An -t x1 -N 12 /tmp/tts-nihao.wav
docker exec lc-l3-test file /tmp/tts-nihao.wav
docker exec lc-l3-test sha256sum /tmp/tts-nihao.wav
```

链路预期：HTTP 200、`content-type: audio/wav`、文件头包含 `RIFF` 和 `WAVE`、格式为 24 kHz 单声道 PCM WAV。

内容预期：当前版本人工试听为乱码，所以 TTS 质量应判定为失败。

## 7. 查看多模态文字并导出音频

以下示例处理视频响应；将路径替换为 `/tmp/mm-image.json` 或 `/tmp/mm-audio.json` 可处理其他用例。

```bash
docker exec lc-l3-test python3 -c '
import base64
import json

response_path = "/tmp/mm-video.json"
wav_path = "/tmp/mm-video.wav"
data = json.load(open(response_path))
message = data["choices"][0]["message"]
raw = base64.b64decode(message["audio"]["data"])

print("text:", message.get("content"))
print("audio_bytes:", len(raw))
print("audio_magic:", raw[:12].hex())
open(wav_path, "wb").write(raw)
print("wav:", wav_path)
'

docker exec lc-l3-test file /tmp/mm-video.wav
docker exec lc-l3-test sha256sum /tmp/mm-video.wav
```

`audio_magic` 应以 RIFF 的十六进制 `52494646` 开头，并在前 12 字节包含 WAVE 的 `57415645`。

## 8. 将 WAV 拉回当前工作机

第一步在远端宿主机执行，将文件从容器复制到远端宿主机：

```bash
docker cp lc-l3-test:/tmp/tts-nihao.wav /tmp/tts-nihao.wav
docker cp lc-l3-test:/tmp/mm-video.wav /tmp/mm-video.wav
```

第二步退出 SSH，在当前工作机执行：

```bash
scp -F /dev/null -i /home/cocoa/.ssh/id_ed25519 -P 22 \
  -o StrictHostKeyChecking=no \
  root@113.46.46.40:/tmp/tts-nihao.wav /home/cocoa/lc/tts-nihao.wav

scp -F /dev/null -i /home/cocoa/.ssh/id_ed25519 -P 22 \
  -o StrictHostKeyChecking=no \
  root@113.46.46.40:/tmp/mm-video.wav /home/cocoa/lc/mm-video.wav
```

本次验证必须人工试听：

- `tts-nihao.wav` 应说“你好世界”；
- `mm-video.wav` 应表达 JSON 中的文字回答；
- 当前阶段预期两项均不能正确表达语义，应记录为失败。

## 9. 验收矩阵

| 用例 | 技术检查 | 内容检查 | 当前预期 |
| --- | --- | --- | --- |
| 纯文字 | HTTP 200、JSON 可解析 | 包含“你好世界” | 通过 |
| 图片 | HTTP 200、JSON 可解析 | 描述多辆汽车 | 通过 |
| 音频 | HTTP 200、JSON 可解析 | 表达询问汽车数量 | 通过 |
| 视频 | HTTP 200、JSON 可解析 | 表达白色触控笔 | 通过 |
| 多模态音频字段 | base64 可解码、WAV 合法 | 与文字回答语义一致 | 链路通过，内容失败 |
| 基础 TTS | HTTP 200、WAV 合法 | 清晰说出“你好世界” | 链路通过，内容失败 |

任何音频用例都必须分别记录“链路结果”和“内容结果”，不能合并成一个“通过”。

## 10. 日志和故障处理

```bash
docker exec lc-l3-test grep -nE \
  'POST /v1/chat/completions|POST /v1/audio/speech|murmur_hash32_kernel|MLIRCompilationError|Failed to compile BiShengLIR|out of memory|Not enough memory|Traceback' \
  /tmp/sgl-omni-run-npu.log | tail -n 150
```

| 现象 | 判断与处理 |
| --- | --- |
| `murmur_hash32_kernel`、BiSheng/MLIR 编译错误 | seeded sampler 规避未生效或请求进入显式 seed 路径；核对代码和请求参数。 |
| `Not enough memory` / OOM | 检查 NPU 0--8 的进程、stage 共置和残留 worker；不要直接通过提高 `mem_fraction_static` 掩盖问题。 |
| HTTP 200 但音频乱码 | 当前已知 TTS 质量问题；不是 WAV 导出路径造成，记录为内容失败。 |
| JSON 没有 `audio.data` | 检查 `modalities`、`audio` 参数和服务日志。 |
| 媒体文字结果错误 | 检查 fixture、预处理 stage、encoder 设备和 Thinker 日志。 |
| 服务端口不可达 | 检查服务进程、端口监听和启动日志末尾。 |

## 11. 资源与设备核对

当前代码仍可能让 image encoder、audio encoder、Code2Wav 在 NPU 环境默认走 CPU。复现报告应同时记录：

```bash
docker exec lc-l3-test npu-smi info
docker exec lc-l3-test ps -eo pid,ppid,%cpu,%mem,args
```

仅看到 Thinker/Talker 占用 NPU，不能证明所有模型组件已经 NPU 化。需要结合启动日志、模型参数设备断言和 CPU 使用率判断。

显存部分至少记录四个时间点：

1. 服务启动前；
2. HCCL 和模型加载后；
3. 完成一轮所有用例后；
4. 服务停止后。

当前约 47 GB/卡只是 TP=8 初始化阶段的未分项显存差值，不能直接全部归因于 HCCL。

## 12. 停止服务和回收检查

先根据进程信息停止本次服务，不要匹配并终止其他任务。若容器专用于本次测试，可用：

```bash
docker restart lc-l3-test
```

随后检查：

```bash
docker exec lc-l3-test pgrep -af 'sgl-omni serve'
docker exec lc-l3-test npu-smi info
```

验收要求：本次服务 worker 全部退出，NPU 0--8 回到启动前基线。若只能通过重启容器回收显存，应在结果中记录为进程生命周期待优化项。

## 13. 结果记录模板

| 项目 | 记录内容 |
| --- | --- |
| 代码 | commit、工作树修改、容器镜像。 |
| 软件 | CANN、torch、torch_npu、SGLang、Transformers。 |
| 资源 | 可见 NPU、stage 卡位、四个时间点的 HBM。 |
| 服务 | 启动命令、端口、就绪时间、日志路径。 |
| 请求 | 完整 JSON、fixture、HTTP 状态、耗时。 |
| 文字结果 | 实际输出、预期语义、是否通过。 |
| 音频链路 | WAV 格式、大小、SHA-256。 |
| 音频内容 | 人工试听、ASR 转写、是否与文字一致。 |
| 异常 | 错误栈、处理方法、对结论的影响。 |

## 14. 当前复现结论

接收方应复现到以下阶段性状态：

- 服务能够启动；
- 文本、图片、音频、视频的文字理解通过；
- 默认无 seed 的 TTS 请求不再触发已知 NPUIR 编译失败；
- TTS 和多模态语音能够生成合法 WAV；
- 音频语义仍为乱码，TTS 质量不通过；
- 部分组件仍可能 CPU 回退，显存占用尚未形成可信分项账单。
