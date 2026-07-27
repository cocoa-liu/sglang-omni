# Qwen3-Omni 多模态输入到语音输出端到端验证（remote_40）

日期：2026-07-22

目标环境：
- 远程配置：`/home/cocoa/lc/scripts/remote_40.json`
- 远程主机：`113.46.46.40`
- Docker 容器：`lc-l3-test`
- 服务：端口 `8000` 上已应用 NPU 默认无 seed 规避的 Qwen3-Omni Speech 服务
- 模型：`/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct`

目标：
- 验证 `/v1/chat/completions` 的图像、音频、视频输入均可在同一次推理链路中进入 Thinker，并在 `modalities=["text", "audio"]` 下经 Talker 和 code2wav 输出语音。
- 每个 case 记录输入媒体、请求参数、HTTP 响应、返回音频格式、服务日志和遇到的问题。

## 步骤 1：确认接口和测试媒体（进行中）

目的：
- 确认 Speech 服务的 OpenAI 兼容接口支持多模态输入和 audio 输出。
- 确认可使用的固定测试媒体，避免下载外部数据。

本地代码依据：
- `ChatCompletionRequest` 支持顶层 `images`、`audios`、`videos` 和 `modalities`。
- `_build_chat_generate_request()` 将这些媒体放入请求 metadata，并将 `modalities` 传入 pipeline。
- 本地仓库测试媒体：`tests/data/cars.jpg`、`tests/data/query_to_cars.wav`、`tests/data/draw.mp4`。

计划操作：
- 只读检查远端代码目录是否已有对应 fixture，并检查补丁服务仍处于就绪状态。
- 对图像、音频、视频分别发送非流式 chat completion 请求，统一设置 `modalities=["text", "audio"]`。
- 视频设置 `video_max_frames=2`，减少视觉 token，避免长视频触发 Talker 上下文长度问题。

观察结果：
- 远端已存在所需 fixture：`cars.jpg`（397 KB）、`query_to_cars.wav`（434 KB）、`draw.mp4`（7.6 MB）。
- `GET /health` 返回 `200`，补丁服务可用于请求验证。

## 步骤 2：图像输入到文字和语音输出（进行中）

请求设计：
- 输入：`tests/data/cars.jpg`。
- 用户指令：简短描述画面中的主要物体。
- 输出：`modalities=["text", "audio"]`、`audio={"voice":"default","format":"wav"}`、非流式。
- 验收：HTTP 200、响应中有 text 和可解码的音频数据、服务日志无 seeded sampler 编译错误。

命令概要：
- 向 `/v1/chat/completions` 发送顶层 `images=["/home/l00951280/sglang-omni/tests/data/cars.jpg"]` 的 JSON 请求。
- 将 HTTP 头保存为 `/tmp/mm_image_headers.txt`，JSON 响应保存为 `/tmp/mm_image_response.json`。
- 使用 Python 仅提取文本、audio base64 解码后的字节数和前 12 字节，不打印整段 base64。

观察结果：
- 返回 `HTTP 200`。
- 文字输出正确描述了四辆不同品牌和颜色的汽车。
- 响应 `message.audio` 存在，解码后为 `70,784` 字节，文件头为 `RIFF...WAVE`。
- usage：`prompt_tokens=6062`、`completion_tokens=48`。

遇到的问题：
- 无。

## 步骤 3：音频输入到文字和语音输出（进行中）

请求设计：
- 输入：`tests/data/query_to_cars.wav`。
- 用户指令：用一句中文说明音频中讨论的内容。
- 输出与验收条件同图像 case：`modalities=["text", "audio"]`、WAV 音频、HTTP 200。

命令概要：
- 向 `/v1/chat/completions` 发送顶层 `audios=["/home/l00951280/sglang-omni/tests/data/query_to_cars.wav"]` 的 JSON 请求。
- 将 HTTP 头保存为 `/tmp/mm_audio_headers.txt`，JSON 响应保存为 `/tmp/mm_audio_response.json`。
- 解析响应的文本、音频字节数和 WAV 文件头。

观察结果：
- 返回 `HTTP 200`。
- 文字输出：`这段音频在询问图片中有多少辆汽车。`，与 fixture 语义一致。
- 响应 `message.audio` 存在，解码后为 `420,644` 字节，文件头为 `RIFF...WAVE`。
- usage：`prompt_tokens=81`、`completion_tokens=11`。

遇到的问题：
- 无。

## 步骤 4：视频输入到文字和语音输出（进行中）

请求设计：
- 输入：`tests/data/draw.mp4`。
- 用户指令：用一句中文说明视频中人物手里拿着什么。
- 输出：`modalities=["text", "audio"]`。
- 为减少视频视觉 token，设置 `video_max_frames=2`；验收条件为 HTTP 200、文字和 RIFF/WAVE 音频均存在。

命令概要：
- 向 `/v1/chat/completions` 发送顶层 `videos=["/home/l00951280/sglang-omni/tests/data/draw.mp4"]`、`video_max_frames=2` 的 JSON 请求。
- 将 HTTP 头保存为 `/tmp/mm_video_headers.txt`，JSON 响应保存为 `/tmp/mm_video_response.json`。
- 解析响应的文本、音频字节数和 WAV 文件头。

观察结果：
- 返回 `HTTP 200`。
- 文字输出：`一个白色的触控笔。`，与现有 video integration fixture 的预期语义一致。
- 响应 `message.audio` 存在，解码后为 `137,684` 字节，文件头为 `RIFF...WAVE`。
- usage：`prompt_tokens=598`、`completion_tokens=7`。

遇到的问题：
- 无。

## 收尾检查

命令概要：
- 检查 `/tmp/sgl-omni-run-npu-unseeded.log` 中最近三条 `/v1/chat/completions` 访问日志。
- 搜索 `murmur_hash32_kernel` 和 `MLIRCompilationError`。

观察结果：
- 图像、音频、视频三次请求均在服务日志中记录为 `HTTP 200`。
- 当前补丁服务日志没有出现 `murmur_hash32_kernel` 或 `MLIRCompilationError`。

结论：
- 已完成图像、音频、视频三种输入分别到文字和 WAV 语音输出的端到端验证。
- 验证经过：输入媒体解析与编码 -> 多模态聚合 -> Thinker 生成文字 -> Talker 生成 codec token -> code2wav 生成音频 -> OpenAI chat completion 响应封装。
- 本次所有 Talker 语音输出均通过 NPU 默认无 seed 规避路径，未触发已知 BiSheng 编译失败。

## 步骤 5：生成语音的独立回读校验（进行中）

目的：
- 将视频 case 生成的 `/tmp/mm_video.wav` 再作为音频输入，要求模型只转写听到的内容，验证输出语音语义与原回答一致。

计划操作：
- 先读取 WAV 的容器内文件信息。
- 向 `/v1/chat/completions` 发送 `audios=["/tmp/mm_video.wav"]` 和 `modalities=["text"]`，不请求新的语音输出。

## 步骤 6：导出图像和音频 case 的语音结果（进行中）

目的：
- 从 `mm_image_response.json` 和 `mm_audio_response.json` 的 base64 audio 字段导出实际 WAV 文件，并传回本机供人工试听。

计划操作：
- 在容器中生成 `/tmp/mm_image.wav` 和 `/tmp/mm_audio.wav`。
- 先导出到 remote_40 宿主机的 `/tmp`，再通过 SCP 传至本机 `/home/cocoa/lc/`。
- 使用 SHA-256 对比远端与本机文件。

观察结果：
- 成功从 JSON 解码出容器内文件 `/tmp/mm_image.wav` 和 `/tmp/mm_audio.wav`。
- 本机文件：
  - `/home/cocoa/lc/mm_image.wav`，约 70 KB，SHA-256 `71cf22e78229d4b143b1315086fae78d905a6f6e1d34ac663ecd0a2055567f30`。
  - `/home/cocoa/lc/mm_audio.wav`，约 411 KB，SHA-256 `57690b283be0153e7768e921d39ffd5b18289ff5e054053ff1c71a7b930bd557`。
- 两个 SHA-256 均与 remote_40 宿主机导出的文件一致。

## 步骤 7：人工听感发现语义失败（进行中）

观察结果：
- 人工试听图像和音频 case 导出的 WAV 后，发现语音混乱，未能表达响应中的文字内容。

影响：
- 之前的 HTTP 200、RIFF/WAVE 文件头和音频字节数只能证明请求链路、codec 和封装未崩溃，不能证明 TTS 语义质量。
- “NPU 默认无 seed 规避恢复 TTS”的结论修正为“恢复了音频产出，不代表语义可用”。

后续定位计划：
- 先将当前生成语音回送为音频输入做独立转写，记录机器可识别的内容。
- 对比 NPU 无 seed 随机采样与 greedy 采样。greedy 路径使用 `argmax`，可绕开 seeded hash 内核和随机采样本身，从而区分“编译规避”与“采样质量”问题。

首次回读尝试：
- 计划向 `/v1/chat/completions` 提交 `/tmp/mm_video.wav` 并只请求 text 转写。
- 实际结果：`curl: (7) Failed to connect to 127.0.0.1 port 8000`，未执行模型推理，未产生转写结果。
- 处置：先检查服务进程和启动日志，确认服务状态后再继续质量定位。

服务状态检查：
- 随后检查确认服务进程已经退出；日志以 `Shutting down` 结束，没有显示模型执行异常。
- 因此回读失败由服务停止导致，不作为语音质量结论的证据。

## 步骤 8：Greedy Talker A/B 验证（进行中）

目的：
- 对比当前 NPU 无 seed 随机采样与不使用 sampler 的 greedy `argmax` 采样，定位混乱语音是否由随机采样路径造成。

修改边界：
- 仅在 `Qwen3OmniTalker` 运行于 NPU 时，令 Talker codec token 选择走 `torch.argmax`。
- CUDA/CPU 路径和既有 sampler 保持不变；远端当前 `talker.py` 将先备份。

执行计划：
- 先检查 NPU 0-8 是否可安全启动服务。
- 备份并应用 greedy 测试补丁，启动原参数服务。
- 对固定文字输入和视频输入重放请求，人工试听并尝试音频回读转写。

首次补丁尝试：
- 本地 `apply_patch` 的路径少写了 `sglang-omni/` 目录，工具在读取文件阶段失败，未修改任何文件。
- 已使用正确路径重试并应用局部 greedy 分支；后续远端修改前会创建当前无 seed 版本的独立备份。

执行结果（部分）：
- NPU 0-8 空闲后，greedy 补丁通过远端 AST 语法检查，服务以原参数成功启动。
- 基础请求 `input="你好世界"` 返回 HTTP 200，生成 `/tmp/qwen3_omni_tts_greedy.wav`：24 kHz 单声道、约 1.234 秒、`x-completion-tokens=48`。
- 回读请求未留下 `/tmp/tts-greedy-readback.json`；在提取转写时发现文件不存在，需检查服务日志后才能给出语义结论。

回读跟踪：
- 服务日志显示回读请求已提交到 preprocessing，但没有后续 decode 或 HTTP 完成记录，响应文件始终未创建；该请求目前卡住。
- 因此未将模型回读当作质量结论。已把 greedy 生成的原始 WAV 传回本机供人工试听：`/home/cocoa/lc/qwen3_omni_tts_greedy.wav`，SHA-256 `aaf5dbc62dd6a8e22fec6c53ba65b6fafe0214f65596bd3352a7c01abf560129`，与 remote_40 一致。

人工听感结果：
- greedy WAV 也不是“你好世界”，而是无意义乱码。

结论更新：
- Greedy 与无 seed 随机采样均生成乱码，采样器不是语义失败的唯一根因；不能以替换 sampler 作为修复方案。
- 后续优先检查 Talker logits、code predictor 位置编码和 codec token 到 code2wav 的接口，而不是继续调节 top-k/top-p 或 seed。

## 2026-07-22: residual codec sampler parity attempt

### Trigger

The unseeded SGLang workaround and the layer-0 greedy workaround both returned valid
24 kHz WAV files, but manual listening found that neither audio intelligibly spoke
the requested text. Therefore a successful HTTP response or RIFF header is not a
functional TTS pass.

### Investigation

1. Read the installed reference implementation in the remote_40 container:
   /usr/local/python3.11.15/lib/python3.11/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py.
2. Its Talker generation calls the residual code predictor with
   do_sample=True, top_k=50, top_p=0.8.
3. The SGLang-Omni implementation instead used argmax for every residual code
   group. This is not equivalent and can produce code sequences outside the
   vocoder's expected generation distribution.
4. Verified an NPU-safe replacement with a small tensor on npu:8:
   torch.multinomial(torch.softmax(torch.randn(2, 2150, device="npu:8"), -1), 1)
   completed successfully. This avoids the failing SGLang/Triton seeded hash
   kernel; it is not the NPUIR path that failed earlier.

### Change under test

Changed only the residual code predictor sampler in the remote container:

- retain the top 50 logits;
- apply nucleus filtering at cumulative probability 0.8;
- call native torch.multinomial(..., 1).

Backup: talker.py.bak-20260722-before-residual-sampler.

The main layer-0 SGLang sampler remains on the existing NPU workaround because its
seeded path still fails during BiSheng compilation. This residual sampler is a
separate correction required for reference parity.

### Service restart issue

Stopping the original launcher PID did not release port 8000 because child
processes remained alive. The replacement service therefore started on port 38767.
This is a test-environment process-lifecycle issue, not a model result. The patched
service reached Application startup complete on 38767.

### Result and required manual check

POST /v1/audio/speech for 你好世界 on the patched service returned HTTP 200 and
produced a valid 24 kHz mono WAV (14294 bytes):

- container: /tmp/qwen3_omni_tts_residual_sampler.wav
- local export: /home/cocoa/lc/qwen3_omni_tts_residual_sampler.wav
- SHA-256: 342a55897c5a30b9f24b7575f29528dcca543f40840cd40ccefd7d0a05fca2d2

This file must be listened to before declaring the fix successful. The expected
content is an intelligible rendering of 你好世界; file validity alone is not
evidence of semantic correctness.

### Follow-up listening result and environment reset

Manual listening of `/home/cocoa/lc/qwen3_omni_tts_residual_sampler.wav` found it
was still unintelligible rather than `你好世界`. Therefore matching only the residual
codec sampler is necessary but insufficient; do not mark this attempt as a TTS fix.

The prior launcher shutdown left orphan multiprocessing workers that retained all
NPU memory. The approved recovery is `docker restart lc-l3-test`, which clears the
test container's complete process tree before the next clean service startup.

## 2026-07-22: both codec levels use native NPU sampling

### Why this attempt was needed

The residual-only sampler fix remained unintelligible on manual listening. Code
inspection then found the layer-0 Talker request uses temperature=0.9, top_k=50,
top_p=1.0, and repetition_penalty=1.05, while the prior NPU workaround forced
layer 0 to argmax. This also changes the checkpoint's intended generation
distribution.

### Change

On NPU only, layer 0 now applies its existing repetition/suppression masks, then
uses native PyTorch temperature scaling, top-k/top-p filtering, and multinomial.
This avoids SGLang's seeded Triton murmur-hash kernel without using greedy decode.
Residual groups retain the earlier reference top_k=50/top_p=0.8 multinomial rule.
Backup: talker.py.bak-20260722-before-layer0-native-sampler.

### Clean-environment verification

`docker restart lc-l3-test` removed orphan stage workers left by prior launcher
shutdown. The new service started cleanly on port 8000 and all 15 stage processes
became ready. A request to `/v1/audio/speech` with input `你好世界` returned HTTP
200 and generated:

- container: `/tmp/qwen3_omni_tts_native_samplers.wav`
- local: `/home/cocoa/lc/qwen3_omni_tts_native_samplers.wav`
- format: 24 kHz, mono, PCM WAV
- bytes: 935024
- SHA-256: `128a229865de54be8f3546856d5c6fc531d7a244d48fd18a8c1cd5ca70020aff`

Status: pending mandatory human listening. The expected result is intelligible
speech of `你好世界`; do not equate HTTP success or duration with a functional fix.

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
