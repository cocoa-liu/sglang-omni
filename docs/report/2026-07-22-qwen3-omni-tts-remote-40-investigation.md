# Qwen3-Omni TTS NPU/NPUIR 问题调查记录（remote_40）

日期：2026-07-22

目标环境：
- 远程配置：`/home/cocoa/lc/scripts/remote_40.json`
- 远程主机：`113.46.46.40`
- Docker 容器：`lc-l3-test`
- 模型路径：`/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct`

目标：
- 严格按 `ascend-a3-manual-verification.md` 复现 Speech/TTS 的 500 错误。
- 用实际日志定位具体失败图和触发条件，避免将“CANN 不支持 Talker 融合算子”作为未经验证的结论。
- 每次尝试记录命令概要、观察结果、遇到的问题和处置。

## 步骤 1：基线复现前的环境检查

目的：
- 确认 `lc-l3-test` 存在、NPU 显存满足 Speech 模式所需的 9 张卡（每卡至少约 55 GB 空闲）。
- 查找是否已有 `sgl-omni` 服务或占用端口 `8000` 的进程，避免干扰现有任务。

计划操作：
- 只读检查 Docker 容器状态、NPU 数和显存、服务进程及端口监听状态。
- 不执行验证文档中的 `docker restart` 或 `pkill`，因为这些会中断可能属于其他人的任务；只有资源确实空闲后才启动新的复现服务。

命令概要：
- 首次使用内嵌 Python 的 SSH 命令读取 `torch_npu.npu.mem_get_info()`。
- 改用 `docker exec lc-l3-test npu-smi info` 检查 NPU 使用情况。
- 通过 `ps -fp` 检查 NPU PID 及其父进程命令。

观察结果：
- 容器 `lc-l3-test` 中已有 Speech 服务，父进程 PID `231144`：
  `sgl-omni serve --model-path /home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct --host 0.0.0.0 --port 8000 --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 --talker-gpu 8 --mem-fraction-static 0.75`
- NPU 0-8 已由该服务占用；NPU 0-7 的进程各约 47 GB，NPU 8 的 Talker 进程约 47 GB。NPU 9-15 空闲。
- 该服务的参数与手册的 Speech 基线完全一致，因此直接对其发起最小 TTS 请求可完成等价复现，且不需要重启容器或启动第二个实例。

遇到的问题：
- 第一次环境检查命令因 SSH/远端 shell/Python 三层引号转义不正确而在 Python 解析阶段失败：`SyntaxError`。

解决方式：
- 改用无嵌套脚本的 `npu-smi info` 成功完成检查。该问题未触及 NPU 或模型服务。

## 步骤 2：对已运行的 Speech 服务做最小 TTS 基线请求

目的：
- 使用手册中的 `/v1/audio/speech` 请求复现端到端失败。
- 记录 HTTP 状态、响应体及服务日志中的首个异常和底层编译报错。

命令概要：
- 在容器内向 `http://127.0.0.1:8000/v1/audio/speech` 发送：
  `{"model":"default","input":"你好世界","voice":"default"}`。
- 将响应体保存到容器的临时文件 `/tmp/qwen3_omni_tts_response.bin`，并读取 HTTP 头。
- 只读提取 `/tmp/sgl-omni-run.log` 中 `murmur_hash32` 前后的日志和 Python 调用栈。

观察结果：
- 请求稳定返回 `HTTP/1.1 500 Internal Server Error`。
- 响应体和服务日志均显示同一个底层错误：
  `In function: murmur_hash32_kernel`
  `fatal error: error in backend: Cannot select: i64 = fp_to_uint`
  `Failed to compile BiShengLIR to binary` / `Failed to compile HIVM IR`。
- 端到端调用栈为：
  `TalkerModelRunner.execute`
  -> `BaseModelRunner._sample_next_token_ids`
  -> `sglang ModelRunner.sample`
  -> `Sampler._sample_from_probs`
  -> `top_k_top_p_min_p_sampling_from_probs_torch`
  -> `multinomial_with_seed`
  -> `murmur_hash32`
  -> `murmur_hash32_kernel`（Triton/BiSheng）。
- 因此失败发生在 Talker 计算出 logits 之后的采样阶段；它不是泛化的“Talker 融合算子不支持”。

遇到的问题：
- 手册将根因概括为“Talker 融合算子组合不支持”，会把排查方向引向模型前向图，和实际栈不符。

解决方式：
- 以本次端到端调用栈为准，将后续定位限定为 SGLang seeded sampler 的 Triton hash 内核和其 seed 注入路径。

## 步骤 3：核对运行中代码的 seed 注入和 sampler 配置

目的：
- 验证端到端请求为什么必然进入 `multinomial_with_seed`。
- 找出可在项目层修复、且不需要等待 CANN 修复的最小边界。

命令概要：
- 只读检查 remote_40 运行中代码：`request_builders.py` 1062-1070、`talker.py` 925-939、1160-1185。
- 对照本地源码中的 Talker stage `sampling_backend` 默认值。

观察结果：
- `request_builders.py` 会在请求未给 seed 时，基于 `request_id` 写入 seed。
- 即使 request builder 不写 seed，`talker.py` 也会把空 seed 转换成由 request id 派生的 rank-shared seed。
- `talker.py` 始终构造 `SamplingBatchInfo(... sampling_seed=self._sampling_seeds[:batch_size])`，并固定 `need_top_p_sampling=True` 与 `need_top_k_sampling=True`。
- Qwen3-Omni Talker stage 默认设置 `sampling_backend="pytorch"`。

当前结论：
- 对当前 NPU 实现，任意正常 TTS 请求都会进入带 seed 的 PyTorch sampler；这条路径必然触发有缺陷的 `murmur_hash32_kernel` 编译。
- 根因由两个层次组成：直接故障是当前 Triton/torch_npu/CANN-BiSheng 不能编译该 hash 内核；项目侧放大因素是 Talker 为全部请求强制构造 seed，使其没有无 seed 的随机采样退路。

建议的验证顺序：
- 先做不改依赖的 A/B：仅在 NPU Talker 上令 `SamplingBatchInfo.sampling_seed=None`，保持现有 `sampling_backend="pytorch"`，重启测试服务后发送同一 TTS 请求。预期：绕过 `multinomial_with_seed`，500 消失。
- 若需要可复现性，再单独验证 `sampling_backend="ascend"`；此前的小张量测试表明其 BF16 logits 的 `top_ps` 和 `min_ps` 必须同为 BF16，否则 `npu_top_k_top_p` 会因 dtype 不一致失败。
- 最后才做依赖升级矩阵（torch_npu、Ascend Triton、CANN），将 `murmur_hash32_kernel` 的最小 repro 作为上游缺陷用例。该路线恢复 seeded sampling，但不应作为阻塞 TTS 可用性的唯一方案。

## 步骤 4：验证 NPU Talker 默认请求不传 sampling seed（进行中）

目的：
- 验证项目层规避：对 NPU Talker 的默认无 seed 请求，不向 SGLang sampler 传入 `sampling_seed`，从而绕过 `multinomial_with_seed -> murmur_hash32_kernel`。

修改边界：
- 修改 `talker.py` 的静态 `SamplingBatchInfo` 构造：当运行设备为 NPU 且该请求没有显式 seed 时，`sampling_seed=None`。
- 本次测试只覆盖 OpenAI Speech API 的默认请求，不改变 CUDA/CPU 逻辑，不更改 CANN、torch_npu 或 SGLang 依赖。
- 显式 seed 的 API 语义需在后续补充“当前 NPU 栈不支持”的清晰报错；本轮不静默丢弃用户 seed。

执行保护：
- 先在远端创建带时间戳的原文件备份，再应用最小补丁。
- 当前服务占用 NPU 0-8，无法并行启动第二个 9 卡实例；因此将只重启该已确认用于验证的服务，启动参数和日志路径保持不变。
- 验证完成后保留备份、补丁后的日志和 HTTP 结果，便于回滚和复核。

### 步骤 4.1：应用补丁并重启测试服务

命令概要：
- 本地补丁通过 AST 语法解析；`py_compile` 因源码目录只读、无法创建 `__pycache__` 失败，改用不写文件的 `ast.parse()` 后通过。
- 远端先备份两份文件到：
  - `request_builders.py.bak-20260722-npu-unseeded`
  - `talker.py.bak-20260722-npu-unseeded`
- 应用最小补丁并在远端再次做 AST 解析，结果通过。
- 向旧服务主进程 PID `231144` 发送 `SIGTERM`，随后用完全相同的模型和启动参数启动补丁服务，日志写入 `/tmp/sgl-omni-run-npu-unseeded.log`。

观察结果：
- 补丁服务在加载阶段失败，尚未开始 TTS 验证。
- 首个失败为 `thinker_tp0` 启动时的：`RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.`
- `npu-smi info` 显示旧服务的 worker PID `232439` 至 `232453` 仍占用 NPU 0-8；这些 PID 均在此前确认属于 PID `231144` 的旧 Qwen3-Omni 服务。主进程退出没有级联结束其 multiprocessing 子进程，造成 NPU 显存残留。

遇到的问题：
- 初次只终止服务主进程不足以回收 NPU worker 显存，导致补丁服务无法启动。

解决方式（进行中）：
- 只向已确认属于旧验证服务的精确 worker PID 发送 `SIGTERM`，随后重新检查 NPU 显存。
- 不会对容器内其他未知进程或 NPU 9-15 的资源执行清理。

### 步骤 4.2：清理旧 worker 后重启并回归 TTS

命令概要：
- 向已确认的旧服务 worker PID `232439` 至 `232453` 发送 `SIGTERM`。
- `npu-smi info` 确认 NPU 0-8 已无运行进程、显存恢复到约 2.8-3.1 GB 基线占用。
- 用与基线完全相同的参数重新启动补丁服务。
- 服务显示 `Application startup complete` 后，重放基线请求：
  `POST /v1/audio/speech`，body 为 `{"model":"default","input":"你好世界","voice":"default"}`。

观察结果：
- 补丁服务正常启动；Thinker 和 Talker 都完成权重加载。
- 相同 TTS 请求返回 `HTTP/1.1 200 OK`，而基线为 `HTTP 500`。
- 响应头：`content-type: audio/wav`，`content-length: 151934`，`x-completion-tokens: 48`。
- 生成文件 `/tmp/qwen3_omni_tts_response_npu_unseeded.bin` 的前 12 字节为：
  `52 49 46 46 76 51 02 00 57 41 56 45`，即 `RIFF....WAVE` 文件头。
- 新服务日志只有该请求的 `200 OK`；未出现 `murmur_hash32_kernel` 或 `MLIRCompilationError`。

遇到的问题：
- 容器内没有 `file` 命令，不能用它验证音频格式。

解决方式：
- 改用 `od` 检查文件头，确认是 RIFF/WAVE。

结论：
- 已验证 NPU Talker 默认无 seed 的项目层规避可恢复端到端音频产出：请求返回 200 且生成有效 WAV。
- 后续人工试听发现语音语义混乱，因此该规避不能视为可用 TTS 修复；需要通过 greedy 与随机采样的 A/B 继续定位。
- 该补丁绕开了有故障的 seeded PyTorch sampler，不依赖 CANN 升级。
- 限制：当前补丁对 NPU 上的显式 `seed` 抛出异常；应在 API 参数校验阶段把它转成清晰的 4xx 响应，并添加相应测试，避免在调度阶段返回 500。

收尾检查：
- 删除了 `request_builders.py` 中已不再引用的 `MAX_INT32_POSITIVE` 常量；该清理不改变运行行为，也不需要重启已通过验证的服务。
- 本地修复文件通过 AST 语法解析，`git diff --check` 通过。

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


## 2026-07-22：官方 Talker 重放与完整 codec trace

### 目标

全序列 Code2Wav 隔离测试经人工试听后仍为乱码。本轮用于确认：在排查后续
自回归帧和 residual code group 前，首个 Talker 前向是否已与官方 Transformers
实现发生偏离。

### 过程与遇到的问题

1. 通过一次性 SGLang trace 捕获 `你好世界` 请求的真实 prefill 输入：16 个宽度
   为 1024 的 BF16 投影 embedding，以及 SGLang 的 layer-0 code 和 Talker hidden state。
2. 首次官方重放直接以 checkpoint key 配合 `strict=False` 加载，结果无效。原因是
   checkpoint 将每个 MoE expert 分别存为 `gate_proj`、`up_proj` 和 `down_proj`，而
   已安装的 Transformers 模块需要打包后的 `gate_up_proj` 和 `down_proj`；直接加载会
   遗漏 expert 权重，因此丢弃该次数值结果。
3. 修正方式为：每层 128 个 expert 的 `gate_proj`、`up_proj` 沿第 0 维拼接为
   `(128, 768, 1024)` 的 `gate_up_proj`，并将 `down_proj` 堆叠为
   `(128, 1024, 384)`。修正后的官方 Talker 加载结果为 `missing_count=0`、
   `unexpected_count=0`。
4. 对同一份 captured prefill 输入，官方 codec-head 的 top-1 为 1049，与 SGLang
   第一帧的 layer-0 code 1049 完全一致。最终 hidden state 不是逐位一致
   （`max_abs=6.0`、`mean_abs=1.358`），因此仍需直接捕获后续 logits；但该结果已排除
   首步主 Talker 或权重加载整体失效。
5. 增加了仅诊断用途的 hook，用于记录每一帧输出的 16 路 codec code；它不修改采样、
   feedback 或 Code2Wav。首次服务启动因多层 shell 引号解析失败而未启动服务，随后将
   启动命令改为单层远程 `bash -lc` 后恢复正常。
6. 控制组非流式请求返回 HTTP 200、286934 字节 WAV，并捕获 76 帧 shape 为 `(16,)`
   的 codec code。第一帧为 `[1049, 1700, 1626, 546, ...]`。

### 当前结论

seeded sampler 的 NPUIR 报错已被绕过，且主 Talker 首步的 layer-0 选择与官方一致。
因此重点转向 residual code predictor 和后续 Talker AR feedback。未设 seed 时不能直接
比较随机采样的 code；后续诊断必须比较同一首帧下的 residual 原始 logits 或分布。

## 2026-07-22：residual predictor 数值对比，根因范围收敛

### 控制变量对比

服务为控制组 `你好世界` 请求的首个 codec 帧捕获了 15 组 residual predictor 原始
logits。通过 `docker restart lc-l3-test` 释放 NPU 8 后，官方 Transformers 重放以已验证的
MoE 打包方式完整加载 Talker 权重，并使用捕获的 SGLang Talker hidden state、layer-0 code
以及每个实际输出的前序 residual code 作为相同前缀。

本过程修正了两个诊断脚本问题：容器重启会清空容器内 `/tmp`，因此重放脚本不能依赖
存放在其中的辅助脚本；官方 `lm_head` 返回一个 tensor，而 SGLang 层返回两个值。两项问题
均未修改模型代码或生成音频。

### 结果

15 组官方/SGLang residual 原始 logits 均显著偏离：绝对均值差为 2.74--11.05，最大差为
26.39--44.44，且全部 top-1 code 不同。例如第 1 组官方为 316、SGLang 为 1700；第 2 组
官方为 1166、SGLang 为 1626。

### 结论与修复方向

该结果排除了 WAV 序列化、Code2Wav 分帧、seeded sampling 与采样随机性导致乱码的可能，
同时确认主 Talker 的首个 layer-0 选择与官方一致。当前根因集中在 SGLang 的增量 residual
predictor 实现，优先怀疑 cached attention、RoPE/position 处理或 fused 权重映射。

不要继续调节 sampler 参数。应以官方 code predictor forward 为基准验证或替换
`_predictor_forward_one_token`：先比较双 token 前缀（`talker_hidden`、layer-0 codec embedding），
再比较每一个 residual token 追加后的结果。在恢复端到端 TTS 验收前，增加覆盖 15 组 logits
的确定性数值测试。


## 2026-07-23：尝试生成官方 Transformers 参考音频

### 目的

用户要求先试听官方实现对“你好世界”的端到端输出。此测试必须使用
`Qwen3OmniMoeForConditionalGeneration.generate(return_audio=True)`，覆盖官方 Thinker、
Talker 和 Code2Wav，而不是此前仅用于数值对比的 Talker 重放。

### 首次尝试与阻塞点

构造文本请求“请直接说：你好世界。不要添加任何其他内容。”后，以 BF16 和
`device_map="auto"` 加载完整官方模型，准备跨空闲 NPU 自动分片。该尝试在模型权重加载前
失败：容器没有安装 `accelerate`，Transformers 明确报错“Using a `device_map` requires
`accelerate`”。没有生成音频，也没有修改模型或服务代码。

### 后续处理

安装 `accelerate` 后重试同一官方生成脚本。若自动分片仍不支持 NPU，则转为官方
Transformers 的分布式张量并行加载方案；不得将 SGLang 生成的音频误称为官方参考音频。


### 安装依赖后的第二次尝试

在容器内安装 `accelerate==1.14.0` 后，官方完整模型能够开始加载。`device_map="auto"`
将模块自动分配到逻辑设备 0--13，并成功完成 2034 个权重张量的加载；但当前验证环境按
既有 SGLang 配置仅应使用 NPU 0--8。生成进入 Talker residual predictor 的 attention mask
计算后，在 NPU 上触发 vector-core / MTE address 异常，未生成 WAV。

该异常发生在官方 Transformers 的自动跨卡分片路径，不能作为官方模型生成内容的结果，也
不能用于判断 TTS 质量。后续改用显式设备映射，将全部模块限制到 NPU 0--8；如官方
Accelerate hook 仍不能正确支持 NPU 跨卡执行，则需要使用官方 Transformers 的 HCCL
分布式张量并行加载，而不是继续使用 `device_map="auto"`。


### 设备范围修正与第三次尝试

资源检查表明主机共有 16 张 NPU；其中 13--15 被其他任务占用，0--12 空闲。此前“仅应使用
0--8”的说法不准确，已更正。第三次尝试通过 `max_memory={0..12: "58GiB"}` 明确限制
官方自动分片只使用 0--12，完整模型再次成功加载。

生成仍在同一代码路径失败：`talker.generate` 调用官方 residual predictor 的
`code_predictor.generate` 后，在 causal attention mask 计算/执行期间触发 NPU vector-core
异常（MTE DDR address out of range，运行时错误 507035）。因此该阻塞不只是占用 NPU 13--15
造成的自动映射问题，而是当前 Transformers + torch_npu + Accelerate 跨卡执行官方完整 TTS
的兼容性问题。

结论：截至本次尝试，未能从该 NPU 容器导出官方完整端到端 WAV，不能提供“官方音频”试听。
官方 Talker 的单卡数值重放可以运行，但完整官方 TTS 需要另一个已验证支持 HCCL 分布式
Transformers 的环境，或先修复该官方 residual predictor attention-mask 的 NPU 运行错误。
