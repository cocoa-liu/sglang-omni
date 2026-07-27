# Qwen3-Omni 昇腾 NPU 适配与验证交付总结

> 交付结论：Qwen3-Omni 已能在昇腾 A3 上启动服务，并完成文本、图像、音频、视频到文字结果的端到端推理；TTS 请求也能完成并返回格式合法的 WAV。当前版本仍属于阶段性适配结果：语音内容为乱码、部分非 AR 模型组件默认降级到 CPU、显存占用尚未完成可信分项核算，因此不满足生产合入标准。

## 1. 目标环境

| 项目 | 内容 |
| --- | --- |
| 设备 | Ascend 910 A3，16 张 NPU，单卡约 61 GB HBM |
| 容器 | `lc-l3-test` |
| 模型 | `Qwen3-Omni-30B-A3B-Instruct` |
| 模型路径 | `/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct` |
| 服务接口 | OpenAI 兼容的 `/v1/chat/completions`、`/v1/audio/speech` |
| 软件基础 | CANN 9.0.0、PyTorch + `torch_npu`、SGLang / SGLang-Omni |

该环境是本次验证基线，不是已经定型的生产配置。正式交付必须另外固定代码 commit、容器镜像和完整依赖版本；若工作树存在未提交修改，仅记录 commit 不足以复现结果。

## 2. 交付成果

### 2.1 NPU 适配基础能力

已完成以下基础适配，使 SGLang-Omni 能识别 NPU、避开 CUDA 专属逻辑，并启动 Qwen3-Omni 的多阶段服务：

- 建立 NPU 设备识别、可见设备解析、显存查询和设备信息查询；NPU 不再依赖 NVML、CUDA SM 版本或 CUDA P2P 判断。
- CUDA 专属的 custom all-reduce、FlashInfer 相关判断在 NPU 上被禁用，使用 NPU/HCCL 可用路径。
- Thinker 和 Talker 的 SGLang 自定义模型在模型配置解析前完成注册，解决 SGLang/Transformers 无法识别 Qwen3-Omni 子模型架构的问题。
- NPU 上禁用 CUDA Graph，避免 CUDA Graph 参数和执行策略进入 NPU 服务。
- 适配 SGLang 版本变化，包括已废弃的 CUDA Graph 参数、`max_total_num_tokens` 属性差异和自定义架构的 `auto_map` 解析要求。
- 增加 NPU 权重加载路径：通过 safetensors 和单 tensor 迁移规避当前 CPU-only PyTorch + `torch_npu` 组合中 `module.to("npu")` 的失败。

### 2.2 服务和多模态理解能力

在 remote_40 环境完成过实际请求验证：

状态口径：

- “通过”：对应能力的技术结果和内容结果均满足当前用例；
- “链路通过”：请求能完成且响应格式正确，但内容质量未通过；
- “未通过”：已获得明确失败证据，不能作为可用能力交付。

| 能力 | 验证结果 | 验收说明 |
| --- | --- | --- |
| 服务启动与健康检查 | 通过 | 服务可在端口 8000 启动，`/health` 返回 200。 |
| 纯文本理解 | 通过 | Thinker 可处理文字请求并返回文字结果。 |
| 图像到文字 | 通过 | `cars.jpg` 返回与图片一致的汽车描述。 |
| 音频到文字 | 通过 | `query_to_cars.wav` 返回“询问图片中有多少辆汽车”的正确语义。 |
| 视频到文字 | 通过 | `draw.mp4` 在 `video_max_frames=2` 时返回“一个白色的触控笔”。 |
| 图像/音频/视频到文字和音频响应 | 链路通过 | 三类请求均返回 HTTP 200、文字和可解码的 RIFF/WAVE 音频字段。 |
| `/v1/audio/speech` | 链路通过 | “你好世界”可返回 24 kHz 单声道 PCM WAV。 |
| TTS 音频语义 | 未通过 | 人工试听为无意义乱码，不能表达输入文字或多模态文字回答。 |

表中的“通过”仅针对对应能力；“链路通过”只表示请求编排、媒体预处理、Thinker 推理、Talker 调用、Code2Wav 调用和响应封装能够完成，不代表语音内容正确。

### 2.3 TTS 编译阻塞的阶段性规避

原始 NPU TTS 会在 SGLang seeded sampler 的 Triton/BiSheng 编译路径失败，典型报错为 `murmur_hash32_kernel`、`Cannot select: i64 = fp_to_uint`、`MLIRCompilationError`。

已验证的临时规避策略如下：

- 对未显式传入 `seed` 的 NPU Talker 请求，不再自动注入 SGLang seeded sampling seed。
- NPU 上使用 PyTorch 原生 top-k、top-p、温度和 `torch.multinomial` 完成采样，避免进入无法编译的 seeded hash 内核。
- NPU 上若显式传入 `seed`，应返回明确的不支持错误，而不是静默改变请求语义。

该策略使默认无 seed 的 TTS 请求不再因上述 NPUIR 编译错误失败，并能产生合法音频文件。它改变了原始 seeded sampler 的实现路径，且显式 seed 仍不受支持，因此只能作为阶段性兼容方案，不能视为正式的采样实现已经完成。

## 3. 当前待优化项

### 3.1 P0：修复 TTS 乱码，恢复语义正确性

#### 当前现状

当前 TTS 处于“链路可完成、内容不可验收”的状态。问题不是 WAV 无法打开，而是进入声码器的 codec code 已与参考计算明显偏离：

- `/v1/audio/speech` 对“你好世界”返回 HTTP 200，并生成 24 kHz 单声道 PCM WAV；图像、音频、视频请求也能在 `modalities=["text", "audio"]` 下返回文字和 WAV 字段。
- WAV 文件头、PCM 参数和字节数均正常，说明 HTTP 响应、codec code 传递、Code2Wav 调用和 WAV 封装没有发生显式异常；这不能证明 Code2Wav 输入或语音内容正确。
- 人工试听“你好世界”及多模态用例导出的 WAV，实际均为无意义乱码，不能表达文字回答。因此 TTS 功能**未通过**，不得因 HTTP 200 或文件可播放而标记为成功。
- 数值定位显示：在相同输入和相同 residual code 前缀下，SGLang residual code predictor 的 15 组原始 logits 与官方 Transformers 参考实现全部明显偏离，绝对均值差为 2.74--11.05、最大差为 26.39--44.44，且 15 组 top-1 全部不同。问题发生在采样之前。

首帧第 0 路 code 曾出现与官方 top-1 同为 1049 的情况，但这只是单一候选编号相同；SGLang 记录的是采样结果，官方记录的是 logits 最大值，且 hidden state 仍存在明显差异。该现象不能证明主 Talker 整体正确。

#### 已完成的尝试

| 尝试 | 做法 | 结果与结论 |
| --- | --- | --- |
| 绕过 seeded sampler 编译失败 | 默认 NPU 请求不自动注入 seed，使用原生 PyTorch top-k/top-p/temperature/`multinomial`，避开 `murmur_hash32_kernel` 的 BiSheng 编译失败。 | 请求不再因 NPUIR 失败，能产生 WAV；但试听仍为乱码。该改动只恢复链路。 |
| layer-0 greedy 对比 | 将第 0 路 code 改为 `argmax`，排除随机采样影响。 | WAV 仍为乱码，说明问题不是随机采样单独造成。 |
| residual sampler 对齐 | residual predictor 使用参考实现的 top-k=50、top-p=0.8 和原生 `multinomial`。 | WAV 仍为乱码，说明仅修正采样策略不足。 |
| Talker feedback 修正 | 后续 Talker 输入改为使用 residual predictor 中间 hidden state，最后一路才使用 residual embedding。 | WAV 仍为乱码。 |
| 全序列 Code2Wav 解码 | 非流式请求累积完整 codec 序列后一次性解码，检查分块窗口拼接是否造成问题。 | WAV 仍为乱码，说明分块 Code2Wav 不是乱码的充分解释；不能单凭该试验证明声码器所有计算均正确。 |
| 官方数值重放 | 保存真实 Talker prefill 输入，单独加载官方 Transformers Talker；再以同一 hidden/code 前缀比较 residual logits。 | 明确 residual predictor 前向计算偏离，根因范围收敛至 cached attention、RoPE/position、KV cache 或 fused 权重映射。 |

#### 后续必须完成的工作

1. 为 `_predictor_forward_one_token` 增加确定性单元测试：分别比较双 token 前缀和每次 residual code 追加后的 attention 输出、每层 hidden state、15 组 logits。
2. 逐项对齐官方 code predictor：RoPE 位置编号、causal mask、KV cache 写入/读取、QKV 布局、GQA repeat、MoE/fused 权重加载。
3. 数值测试通过后，比较第 2、3 帧及后续帧的 layer-0 原始 logits，确认 Talker AR feedback 没有继续偏离。
4. 最后执行“你好世界”和图像/音频/视频固定用例，使用人工试听与独立 ASR 交叉验证音频语义；只有语义与文字回答一致才可关闭此项。

验收标准：

- residual predictor 15 组 logits 与参考实现在约定 BF16 容差内一致，且 top-k 排序满足回归标准；
- 连续多帧 layer-0 和 residual code 未出现首个可定位偏离点；
- 固定 TTS 和多模态用例的人工试听、ASR 转写均与预期文本一致；
- 修复不重新引入 seeded sampler NPUIR 编译失败。

### 3.2 P0：消除关键组件的 CPU 降级

#### 当前具体降级位置

为绕过当前 PyTorch/`torch_npu` 组合中模块递归执行 `module.to("npu")` 时落入 CUDA dispatch 的失败，代码中保留了以下 NPU 默认 CPU 分支：

| 组件 | 当前文件与位置 | 当前行为 | 风险 |
| --- | --- | --- | --- |
| 图像编码器 | `sglang_omni/models/qwen3_omni/components/image_encoder.py` 的 `Qwen3OmniImageEncoder.__init__` | `device is None` 且设备类型为 NPU 时，默认 `device="cpu"`。 | 图片特征提取可能常态运行在 CPU。 |
| 音频编码器 | `sglang_omni/models/qwen3_omni/components/audio_encoder.py` 的 `Qwen3OmniAudioEncoder.__init__` | 同样默认 `device="cpu"`。 | 音频特征提取可能常态运行在 CPU。 |
| Code2Wav 模型加载 | `sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py` 的 `load_code2wav_model` | 未显式传入设备时，NPU 环境默认 `device="cpu"`。 | 神经网络声码器可能在 CPU 执行，影响延迟、吞吐和资源归属。 |
| Code2Wav scheduler 工厂 | 同文件的 `create_code2wav_scheduler` | 未传入 `gpu_id` 时再次默认 `device="cpu"`。 | 即使服务运行于 NPU，Code2Wav 也可能被静默放到 CPU。 |

将最终波形 `.detach().cpu()` 后用于 WAV 序列化属于正常的数据导出，不是需要消除的推理降级；需要消除的是模型权重和前向计算被放到 CPU。

#### 必须如何修改

1. 在 stage 创建处为 image encoder、audio encoder、Code2Wav 始终传入明确的 `npu:<id>`，不能依赖 `device is None` 的默认分支。
2. 删除或改写上述 `device="cpu" if _gdt() == "npu"` 分支：NPU 环境未获得显式设备时应报配置错误，而不是静默回退。
3. 复用并完善 `models/weight_loader.py` 的 safetensors 直接 NPU 加载和单 tensor 参数/buffer 放置，补齐这些组件的 BF16 前向测试；不得重新使用会触发 CUDA dispatch 的模块级 `module.to("npu")`。
4. 在每个 stage 就绪时记录并断言模型参数设备、输入 tensor 设备和输出 tensor 设备；测试中应检查 `next(model.parameters()).device.type == "npu"`。
5. 为 Code2Wav、图像编码器、音频编码器建立独立 NPU 冒烟测试和端到端性能测试。若某个官方算子尚无 NPU 支持，应显式失败并记录缺失算子，不得转 CPU 后继续成功返回。
6. 根据正式卡位将编码器和 Code2Wav 放在专用 NPU 或有明确显存预算的共置 NPU；CPU 只保留请求编排、文件读取和最终 WAV 字节封装。

验收标准：

- 服务启动日志明确给出三个组件的目标 NPU，且模型参数、输入和输出 tensor 的设备断言全部通过；
- 运行图像、音频和 TTS 用例时，CPU 不承担上述模型的主要前向计算；
- 删除默认 CPU 分支后，功能、稳定性和性能回归通过；
- 任一 NPU 算子不支持时服务应明确失败，不得静默降级。

### 3.3 P1：修复 30B 模型不合理的显存占用并形成资源规划

#### 当前现状和异常

验证服务采用 Thinker TP=8（NPU 0--7）和 Talker 单卡（NPU 8）。该布局能够启动，但当前记录不足以解释实际 HBM 去向，且启动参数表现明显异常：

- 30B 参数以 BF16 保存时，参数数据的理论量级约为 60 GB；这是整个 30B 参数集合的粗略基准，不等于 Thinker、Talker、Code2Wav 任一子模块或单个 TP rank 的实际占用。
- 实测 TP=8 初始化前后，每张 61 GB NPU 的可用显存从约 60 GB 降至约 13 GB，存在约 **47 GB/卡** 的未分项差值。该差值发生在 HCCL/模型运行时初始化阶段，但尚无逐项证据证明它全部属于 HCCL 通信缓冲。
- `mem_fraction_static=0.05--0.60` 均无法启动，只有设置到 **0.75** 才能通过当前 KV cache 计算路径。参数含义与实际表现不直观，说明可用显存采样、已用显存归属或 KV cache 预算公式至少有一项需要重新验证。
- 早期 stage 共置还曾使 image encoder、audio encoder、Code2Wav 与 thinker_tp0 落在同一 NPU 0，出现 `NPU out of memory`；即使模型不大，错误的 stage 放置和静态池预留也会导致 OOM。
- 服务停止后曾出现 multiprocessing worker 未退出、NPU 0--8 持续保留显存的情况，下一次启动报 `Not enough memory`。这说明进程生命周期和显存回收也尚未工程化。

因此，当前的 `mem_fraction_static=0.75` 只是能够启动验证服务的经验参数，不应直接作为生产配置。在完成分项测量前，不能把约 47 GB/卡的差值全部归因于 HCCL，也不能证明 30B 模型已被高效分片和调度。

#### 必须如何优化

1. 将显存拆分测量并写入启动日志：HCCL 初始化前后空闲显存、各 TP rank 权重、KV cache、attention workspace、编码器、Code2Wav、Python 进程和其他任务占用。禁止只看单一全局空闲显存差值。
2. 使用 CANN/HCCL 诊断能力核实约 47 GB/卡差值的构成；确认通信缓冲确实异常后，再评估通信参数或 CANN/HCCL 版本升级，避免在没有证据时直接将问题归因于 HCCL。
3. 使用进程级 NPU 显存统计为每个 stage 计算 KV cache headroom；`total_gpu_memory_fraction` 应基于该 stage 已加载内存，而非其他 stage 或其他用户进程造成的全局波动。
4. 固化设备放置：Thinker TP rank、Talker、Code2Wav、图像编码器、音频编码器不得无预算共置；若共置，配置中的总显存比例必须可校验且不超过单卡上限。
5. 在冷启动、首请求、稳定并发和停止服务后分别采集 HBM。停止后应确认 worker 退出、显存回到基线；若无法回收，必须修复 supervisor/worker 的进程组退出逻辑，而不是依赖人工重启容器。
6. 为正式环境确定基线：每卡峰值 HBM、KV cache token 容量、最大上下文、最大并发、首 token 延迟、端到端延迟和吞吐。只有在这些指标符合资源预算后，才能确定最终 TP 数和 `mem_fraction_static`。

验收标准：

- 输出按 NPU、进程和 stage 划分的显存账单，权重、KV cache、通信、workspace 与无法归类项之和能解释 `npu-smi` 观测值；
- TP=8 各 rank 的占用差异在约定范围内，不再出现无预算 stage 共置；
- 启停循环后显存稳定回到基线，不依赖重启容器；
- 最终 `mem_fraction_static`、上下文和并发参数由测量结果推导，并通过压力测试验证。

### 3.4 P1：完善正式依赖和版本治理

当前验证容器使用过 `--no-deps` 安装并补装依赖，也额外安装过 `accelerate` 用于官方参考复现。这适合排障，不适合正式发布。

正式合入需锁定并验证：CANN、PyTorch、torch_npu、SGLang、Transformers、safetensors、音视频依赖和 HCCL 版本；CUDA 专属依赖应移到可选 extra，NPU 安装不应尝试下载 CUDA wheel。交付包还应包含：

- 唯一代码 commit 和干净工作树，或可审计的补丁集；
- 可重复构建的容器镜像及其 digest；
- NPU/CUDA 条件依赖分组和离线安装清单；
- 安装后依赖一致性检查、最小启动测试和固定用例回归。

### 3.5 P2：官方完整 Transformers 参考链路

为定位 TTS 已尝试在 NPU 上运行官方 `Qwen3OmniMoeForConditionalGeneration.generate(return_audio=True)`。模型可加载，但官方 residual predictor 的 attention-mask 路径在当前 Transformers + torch_npu + Accelerate 跨卡执行中触发 NPU vector-core 异常，未能导出官方端到端 WAV。

这不影响已完成的单卡官方 Talker 数值对比，但后续应建设可运行官方完整参考的环境，或实现等价的确定性参考测试，以便长期回归验证。

## 4. 正式交付建议与验收门槛

建议将本次结果定位为“阶段性交付：NPU 服务化、多模态文字理解和 TTS 链路验证完成；TTS 质量、NPU 全量化和显存治理仍待完成”。正式发布前至少满足：

1. 不依赖 CPU 作为关键模型组件的常态推理回退。
2. NPU seeded sampler 或等价方案可稳定运行，且显式 seed 的行为有明确、可复现的产品定义。
3. residual predictor 数值测试通过，和参考实现的误差落在预先定义的 BF16 容差内，top-k 行为一致。
4. “你好世界”、图像、音频、视频三类固定用例的语音均经人工试听或独立 ASR 验证，语义与文字回答一致。
5. 在目标卡位和目标并发下完成稳定性、性能和显存回归，显存分项能够解释 `npu-smi` 观测值。
6. 服务多次启停后 worker 和显存能够自动回收，不依赖重启容器。
7. 代码、镜像、依赖、启动参数和固定测试数据均已版本化。

## 5. 相关文档

- [人工验证手册](ascend-a3-manual-verification.md)：已有人工验证步骤。
- [TTS 流程与术语说明](qwen3-omni-tts-architecture-guide.md)：Talker、codec code、residual predictor 与 Code2Wav。
- [TTS 调查记录](2026-07-22-qwen3-omni-tts-remote-40-investigation.md)：逐次排障证据和根因收敛过程。
- [NPU 适配实现说明](qwen3-omni-ascend-a3-implementation-guide.md)：系统架构、实现策略和关键代码修改。
- [复现与操作手册](qwen3-omni-ascend-a3-reproduction-guide.md)：环境、启动、测试、验收和结果记录。
