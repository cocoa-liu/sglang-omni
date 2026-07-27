# Qwen3-Omni 昇腾 NPU 适配实现说明

## 1. 文档定位

本文解释本次 Qwen3-Omni 昇腾 NPU 适配是如何实现的。阅读对象不需要预先了解 SGLang-Omni；文档先介绍系统架构和请求流程，再说明 NPU 适配策略、具体代码修改及当前实现边界。

本文不承担操作手册职责。环境准备、启动命令和测试请求见 `qwen3-omni-ascend-a3-reproduction-guide.md`；交付状态和待优化项见 `qwen3-omni-ascend-a3-delivery-summary.md`。

## 2. SGLang-Omni 是什么

SGLang 主要解决大语言模型的高性能推理和批处理。SGLang-Omni 在其上增加多模态流水线：一个请求不再只经过单一语言模型，而是由多个 stage 分工处理图片、音频、视频、文字生成和语音生成。

### 2.1 运行时组成

| 组成 | 作用 |
| --- | --- |
| HTTP 服务 | 接收 OpenAI 兼容请求，返回文字、JSON 或 WAV。 |
| Coordinator / 控制面 | 创建请求、选择流水线、维护请求状态并汇总最终结果。 |
| Stage worker | 每个 stage 运行在独立进程中，负责一种模型或处理任务。 |
| Scheduler | 决定 stage 何时取请求、如何组 batch，以及 AR 模型如何管理 KV cache。 |
| Relay / inbox / outbox | 在 stage 之间传输控制消息、媒体特征、hidden state、codec code 和音频片段。 |
| SGLang ModelRunner | 承载 Thinker、Talker 等自回归模型，负责权重、TP、KV cache 和 token 生成。 |

这种拆分的好处是不同 stage 可以使用不同设备、并发策略和调度器；代价是设备放置、进程生命周期、显存预算和跨 stage 数据契约必须保持一致。

### 2.2 Qwen3-Omni 的 stage

```text
HTTP API
  │
  ▼
preprocessing
  ├─ 文字：聊天模板、tokenization
  ├─ 图片：读取、缩放、切分
  ├─ 音频：读取、特征准备
  └─ 视频：抽帧、时间信息准备
  │
  ├──────────────┐
  ▼              ▼
image_encoder  audio_encoder
  └──────┬───────┘
         ▼
    mm_aggregate
         │
         ▼
       Thinker
  多模态理解和文字生成
         │
         ├──────────────► 文字响应
         │
         ▼
      Talker AR
  每帧生成 16 路 codec code
         │
         ▼
      Code2Wav
  codec code → PCM 波形
         │
         ▼
  JSON 中的 base64 WAV
  或直接返回 WAV 文件
```

并非每个请求都会经过所有 stage：

- 只请求文字时，Thinker 生成文字后即可结束；
- 请求语音时，Thinker 的输出和 hidden state 继续进入 Talker；
- `/v1/audio/speech` 走面向 TTS 的请求构建路径，最终同样经过 Talker 和 Code2Wav；
- 图片、音频、视频只有在对应输入存在时才进入各自的预处理和编码 stage。

### 2.3 两类 scheduler

Thinker 和 Talker 是自回归模型，需要逐 token/逐帧生成并维护 KV cache，因此使用 SGLang 的生成调度路径。图像编码器、音频编码器和 Code2Wav 更接近一次前向或流式窗口处理，使用较简单的 scheduler。

该差异直接影响 NPU 适配：

- Thinker/Talker 的设备、TP、显存池由 SGLang ModelRunner 管理；
- encoder/Code2Wav 自己构造 Transformers 模块并加载权重；
- 不能只修复 ModelRunner，就假设所有 stage 已经在 NPU 上执行。

## 3. Qwen3-Omni 模型链路

### 3.1 Thinker

Thinker 接收文字 token 和媒体特征，输出：

- 面向用户的文字回答；
- Talker 需要的 token embedding 和指定层 hidden state；
- 多模态位置、mask 等辅助信息。

本次图像、音频和视频用例的文字结果符合预期，说明预处理、媒体编码、多模态聚合和 Thinker 主链路已经能够完成。

### 3.2 Talker

Talker 不直接输出波形，而是按时间帧生成离散 codec code。每帧包含 16 路：

- 第 0 路由主 Talker 生成，提供该帧的主要声音信息；
- 第 1--15 路由 residual code predictor 依次生成，补充细节；
- 当前帧的 code 和文字侧 hidden state共同构成下一帧 Talker 的反馈输入。

因此第一帧 residual code 算错后，错误不仅影响当前帧，还可能通过反馈影响第二帧及之后的主 Talker 输出。

### 3.3 Code2Wav

Code2Wav 将形状近似为 `[batch, 16, time]` 的 codec code 序列解码为 PCM。PCM 再被封装为 24 kHz 单声道 WAV。

“WAV 文件可解析”只能证明输出格式合法。若 codec code 错误，Code2Wav 仍可能稳定地产生一段可播放但无语义的声音。

## 4. NPU 适配的总体策略

本次适配按四个层次进行，而不是简单将 `cuda` 字符串替换成 `npu`。

### 4.1 能力识别

新增统一设备判断，使代码根据能力选择实现：

- 使用 `torch.npu` 查询设备数量、名称、总显存和进程内分配；
- 使用 `ASCEND_RT_VISIBLE_DEVICES` 解释 NPU 可见卡；
- NPU 不访问 NVML、CUDA SM、CUDA P2P；
- NPU 禁用 FlashInfer 和 CUDA custom all-reduce 等 CUDA 专属路径。

相关文件：

- `sglang_omni/utils/device.py`
- `sglang_omni/utils/gpu_compat.py`
- `sglang_omni/utils/gpu_memory.py`
- `sglang_omni/utils/misc.py`

### 4.2 模型注册和 SGLang 版本兼容

Qwen3-Omni 的 Thinker、Talker 是 SGLang-Omni 自定义架构。当前 SGLang 会在 ModelRunner 初始化前解析 Transformers 配置，因此模型类必须提前注册。

实现方式：

1. 在 `SGLModelRunner._register_omni_model_static()` 中维护自定义架构到 Python 类的映射；
2. `ModelWorker` 构造 `ModelConfig` 前执行注册；
3. 为自定义子配置补充 `auto_map`/`model_impl` 兼容信息；
4. 兼容当前 SGLang 已删除或改名的运行时属性。

相关文件：

- `sglang_omni/model_runner/sglang_model_runner.py`
- `sglang_omni/model_runner/model_worker.py`
- `sglang_omni/scheduling/sglang_backend/server_args_builder.py`

### 4.3 权重加载和设备放置

验证环境中的 PyTorch/`torch_npu` 组合在模块递归执行 `module.to("npu")` 时曾落入 CUDA dispatch。为绕过该问题，实现增加：

1. 从 safetensors 直接读取到目标 NPU；
2. 优先使用 `load_state_dict(assign=True)`；
3. 对参数和 buffer 做单 tensor 的设备与 dtype 放置；
4. 非 NPU 路径继续使用标准 `module.to()`。

主要文件为 `sglang_omni/models/weight_loader.py`。

这只是 NPU 权重加载基础。当前 image encoder、audio encoder 和 Code2Wav 在未显式传入设备时仍有默认 CPU 分支，正式合入前必须消除，详见交付总结。

### 4.4 调度、TP 和显存

NPU 上显式禁用 CUDA Graph，并清理当前 SGLang 不再接受的 CUDA Graph 参数。验证布局为：

| stage | 验证设备 |
| --- | --- |
| Thinker | TP=8，NPU 0--7 |
| Talker | NPU 8 |
| image/audio encoder、Code2Wav | 当前代码仍可能因默认参数走 CPU，必须以启动日志和设备断言核实 |

当前 `mem_fraction_static=0.75` 是能够启动验证服务的经验参数，不是生产基线。TP=8 初始化阶段存在约 47 GB/卡的未分项显存差值，必须完成权重、KV cache、通信和 workspace 的分项核算后再确定资源配置。

相关文件：

- `sglang_omni/models/qwen3_omni/stages.py`
- `sglang_omni/pipeline/stage_workers.py`
- `sglang_omni/pipeline/stage/runtime.py`
- `sglang_omni/scheduling/omni_scheduler.py`
- `sglang_omni/config/topology.py`

### 4.5 TTS sampler 的阶段性规避

原始 seeded sampler 在 NPU 上触发 Triton/BiSheng `murmur_hash32_kernel` 编译失败。阶段性处理为：

- 默认请求不再自动注入 seed；
- NPU 使用 PyTorch 原生 temperature、top-k、top-p 和 `torch.multinomial`；
- 显式 seed 不被静默忽略，应返回明确的不支持错误。

该处理恢复了默认 TTS 请求链路，但没有修复音频语义，也没有提供与原 seeded sampler 等价的可复现行为。

相关文件：

- `sglang_omni/models/qwen3_omni/request_builders.py`
- `sglang_omni/models/qwen3_omni/components/talker.py`

## 5. 具体修改清单

| 修改域 | 文件 | 已实现内容 | 当前性质 |
| --- | --- | --- | --- |
| 依赖隔离 | `pyproject.toml` | 将 CUDA relay、CUDA attention 等依赖移出 NPU 必装路径。 | 应保留，需锁定正式版本。 |
| 设备抽象 | `utils/device.py`、`utils/gpu_compat.py`、`utils/gpu_memory.py`、`utils/misc.py` | NPU 检测、显存、可见卡和 CUDA 能力隔离。 | 基础适配。 |
| 模型注册 | `model_runner/sglang_model_runner.py`、`model_runner/model_worker.py` | 提前注册 Thinker/Talker，自定义配置兼容。 | 基础适配。 |
| 权重加载 | `models/weight_loader.py` | safetensors 直接 NPU 加载和单 tensor 放置。 | 已实现基础，需覆盖 encoder/Code2Wav 测试。 |
| stage 启动 | `models/qwen3_omni/stages.py`、`pipeline/stage_workers.py`、`pipeline/stage/runtime.py` | NPU 设备设置、禁用 CUDA Graph、stage 初始化。 | 已验证服务可启动。 |
| 调度和显存 | `scheduling/*`、`config/topology.py` | SGLang API 兼容、显存预算和共置约束。 | 显存账单仍不完整。 |
| 通信 | `relay/nccl.py`、`relay/shm.py` | 避免 CUDA/NCCL 专属假设，适配 NPU 数据传递。 | 需性能和压力测试。 |
| 媒体组件 | `components/image_encoder.py`、`audio_encoder.py`、`preprocessor.py`、`preprocessing/video.py` | 媒体模型和预处理的设备兼容。 | 仍有 CPU 默认回退。 |
| TTS | `components/talker.py`、`request_builders.py`、`components/code2wav_scheduler.py` | sampler 规避、Talker/residual 调试、Code2Wav 调度。 | 链路通过，语义未通过。 |

## 6. TTS 问题是如何定位的

定位遵循“先确认输出层，再向模型内部收敛”的顺序：

1. 检查 HTTP、WAV 头、PCM 参数和音频长度，确认服务能够完成。
2. 人工试听确认“你好世界”为乱码，否定“文件可播放即成功”。
3. 对比 greedy、原生随机采样和 residual sampler，确认乱码不是单一采样参数造成。
4. 使用全序列 Code2Wav 解码检查分块拼接；结果仍乱码，说明分块策略不是充分解释。
5. 捕获真实 Talker prefill 输入、hidden state、首帧 code 和完整 codec 序列。
6. 将相同输入送入官方 Transformers 单卡 Talker 做局部参考前向。
7. 固定 Talker hidden、layer-0 code 和前序 residual code，对比 15 组 residual logits。

结果是 15 组 residual logits 均显著偏离，且 top-1 全部不同。由于差异发生在采样前，当前优先检查：

- residual predictor 的 position/RoPE；
- causal mask；
- KV cache 写入和读取；
- QKV/GQA 布局；
- MoE/fused 权重映射。

首帧第 0 路 code 与官方 top-1 曾同为 1049，但这只是单个编号相同，且两侧比较对象分别是采样结果和 logits 最大值，不能据此证明主 Talker 正确。

## 7. 正式合入前的工程化要求

1. 删除 image encoder、audio encoder、Code2Wav 的 NPU 默认 CPU 回退，并增加设备断言。
2. 修复 residual predictor，建立逐层 hidden/logits 的确定性回归测试。
3. 恢复可定义、可复现的 NPU seeded sampling 行为。
4. 输出按 NPU、进程和 stage 分项的显存账单，解释约 47 GB/卡的初始化差值。
5. 修复 worker 退出和显存回收，服务启停不依赖重启容器。
6. 固化依赖版本、卡位、TP、上下文、并发、性能和稳定性基线。
