# SGLang-Omni 昇腾 A3 NPU 移植 — 端到端验证报告

**日期**: 2026-07-10 ~ 2026-07-15
**目标**: 在昇腾 A3 (910B NPU) 上跑通 sglang-omni + Qwen3-Omni 30B 模型
**输入**: 文字 + 图片 + 音频 + 视频
**输出**: 文字 + 语音（Talker 未通过）
**环境**: 远程 A3 (113.46.38.25), Docker: lc-l3-test

---

## 1. 环境概要

| 项目 | 详情 |
|------|------|
| NPU | Ascend910_9362 (910B), 16 张, 单卡 61GB |
| CANN | 9.0.0 (2026-04-28 build) |
| torch | 2.10.0+cpu + torch_npu 2.10.0 |
| sglang | 0.5.12.post1 |
| sglang-omni | 0.1.0 (editable install) |
| 模型 | Qwen3-Omni-30B-A3B-Instruct BF16, 15 safetensors |

---

## 2. 检查点最终状态

| CP | 检查点 | 状态 | 说明 |
|----|--------|------|------|
| CP1 | Speech 模式启动 (9 Stage 就绪) | ✅ | mem=0.75, ASCEND_AUTO_CONNECT=0 |
| CP2 | 文本输入 → 文本输出 | ✅ | speech/text-only 模式均验证 |
| CP3 | 图片输入 → 文本输出 | ✅ | 正确分析图片内容 |
| CP5 | 音频输入 → 文本输出 | ✅ | 440Hz 正弦波识别为"电子提示音/忙音" |
| CP6 | 视频输入 → 文本输出 | ✅ | torchvision CPU 解码，正确描述颜色交替视频 |
| CP7 | 多模态混合输入 → 文本输出 | ✅ | 文字+图片+音频三模态同时输入，分别描述 |
| CP4 | TTS 语音合成 | ❌ | CANN 9.0.0 BiSheng 编译器多级 bug |
| CP8 | 语音输出（端到端） | ❌ | 依赖 CP4 |

---

## 3. 代码修改清单

### 3.1 sglang-omni 代码（持久化在仓库）

| # | 文件 | 问题 | 修改 | 状态 |
|---|------|------|------|------|
| 1 | `models/qwen3_omni/config.py` | `factory_args` 硬编码 `device="cuda"` | 移除 3 处硬编码，让 stages.py 自动检测 | ✅ |
| 2 | `models/qwen3_omni/stages.py` | encoder factory 在 NPU 上调 `get_device_string(0)` | `device is None` → NPU 用 `"cpu"` | ✅ |
| 3 | `pipeline/stage_workers.py` | `CUDA_VISIBLE_DEVICES` 对 NPU 无效 | 替换为 `ASCEND_RT_VISIBLE_DEVICES` | ✅ |
| 4 | `utils/misc.py` | `avail_gpu_mem()` NPU 返回 None | 改 `torch.cuda.is_available()` → `get_device_type()` | ✅ |
| 5 | `models/qwen3_omni/components/preprocessor.py` | TTS `input` 字符串被当作 `messages` | 字符串自动包装为 `[{"role":"user","content":inputs}]` | ✅ |
| 6 | `models/qwen3_omni/components/talker.py` | `self._device` 未初始化 | `__init__` 添加 `self._device = device` | ✅ |
| 7 | `models/qwen3_omni/components/talker.py` | `self._device` 未初始化、`_predictor_positions` dtype=long | 初始化修复 + 改为 float32 | ✅ |
| 8 | `preprocessing/video.py` | torchcodec 依赖 CUDA + qwen-vl-utils 兼容 | NPU 跳过 + 多版本兼容 | ✅ |
| 9 | `model_runner/base.py` | `torch.device("cuda:N")` 硬编码 | 改用 `get_device_string()` + `set_device()` | ✅ |
| 10 | `model_runner/model_worker.py` | Qwen3OmniTalker 未注册 + max_total_num_tokens 缺失 | transformers 注入 + getattr 回退 | ✅ |
| 11 | `model_runner/sglang_model_runner.py` | 模型注册需在实例化前 | 提取 `_OMNI_MODELS` 类变量 + 静态方法 | ✅ |
| 12 | `pipeline/stage/runtime.py` | device 字符串硬编码 + torch.cuda.set_device | 改用 device 抽象层 | ✅ |
| 13 | `pipeline/runtime_config.py` | `parse_gpu_id()` 只认 "cuda" | 增加 "npu" 支持 | ✅ |
| 14 | `relay/nccl.py` | `dist.init_process_group("nccl")` | 改用 `get_distributed_backend()` | ✅ |
| 15 | `relay/shm.py` | `torch.cuda.synchronize()` | 改用 `synchronize()` | ✅ |
| 16 | `utils/gpu_memory.py` | pynvml API NPU 不兼容 | torch.npu 替换 (4 函数) | ✅ |
| 17 | `utils/gpu_compat.py` | CUDA 专有 API NPU 不兼容 | NPU 守卫返回安全默认值 (5 函数) | ✅ |
| 18 | `models/weight_loader.py` | safetensors `module.to("npu")` 失败 | 直接加载到 NPU 内存 | ✅ |
| 19 | `scheduling/omni_scheduler.py` | sglang 0.5.13 API 兼容 | 实例属性 + 方法 no-op | ✅ |
| 20 | `scheduling/engine_factory.py` | `device="cuda:0"` 硬编码 | 改用 `get_device_string()` | ✅ |
| 21 | `scheduling/generation_batch_policy.py` | CUDA graph 属性访问 | `getattr` 安全访问 | ✅ |
| 22 | `scheduling/sglang_backend/server_args_builder.py` | sglang 0.5.13 移除 cuda_graph 参数 | pop 掉被移除的参数 | ✅ |
| 23 | `models/qwen3_omni/components/*` | 4 个组件 device="cuda" 硬编码 | NPU→CPU 自动回退 | ✅ |

### 3.2 环境变量

| 变量 | 作用 | 必需 |
|------|------|------|
| `ASCEND_AUTO_CONNECT=0` | 修复 HCCL 8-way 建链死锁 | ✅ |
| `HCCL_NPU_SOCKET_PORT_RANGE` | 避免端口冲突 | 推荐 |
| `OMP_NUM_THREADS=4` / `OPENBLAS_NUM_THREADS=4` | CPU 视频帧处理线程限制 | CP6 需要 |

---

## 4. CP4 TTS 阻塞分析

### 4.1 TTS 处理流程

```
POST /v1/audio/speech {"input":"你好世界"}
  → preprocessor: 字符串→messages ✅
  → thinker (8×TP NPU): 文本理解→生成文字 ✅ (首次对话已缓存编译)
  → talker_ar (SGLang ModelRunner): 文字→音频 codec token
    → code_predictor 自回归采样
      → torch_npu inductor 自动拦截（首次运行，无缓存）
        → triton.compile() → BiShengHIR → HIVM → ❌
```

### 4.2 Thinker 为什么不触发

Thinker 首次文本对话时也经过 inductor 编译链，但其算子组合（matmul + softmax + layernorm + MoE）CANN 编译器能处理，编译成功并缓存。Talker 的 **code_predictor 自回归采样路径**（argmax + gumbel noise + uint32 cast + log + clamp）产生了新算子组合，CANN 处理不了。

### 4.3 结论

CANN 9.0.0 BiSheng 编译器链存在多级 bug，torch_npu inductor 自动编译机制不可关闭，Talker 的算子组合无法编译通过。需等待华为发布新版 CANN。已尝试的规避方案全部失败，详见 CP4 专题记录。

---

## 5. 问题与修复详解

### P1: Encoder 组件在 NPU 上强制使用 CUDA 设备

**现象**: 启动 speech 模式时，image_encoder、audio_encoder、code2wav 三个 stage 尽管模型加载到了 CPU，相关进程仍然调用 `torch.cuda.set_device()`，在 NPU 上无效。日志中能看到的直接后果是 encoder 在 NPU 0 上与 thinker_tp0 争抢显存，导致 thinker_tp0 OOM（`NPU out of memory. Tried to allocate 386.00 MiB`）。

**分析**: 在 `config.py` 的三个 stage 工厂函数中，`factory_args={"device": "cuda", ...}` 被直接传入。Pipeline 框架拿到这个值后，原封不动传递给 stage 工厂函数（如 `create_image_encoder_executor`）。工厂函数把 `device="cuda"` 传给模型构造函数 `Qwen3OmniImageEncoder(model_path, device="cuda")`。模型 `__init__` 里的 `if device is None: device = "cpu" if is_npu() else "cuda"` 这段逻辑被绕过了——因为 `device` 是 `"cuda"`，不是 `None`。

**为什么要 CPU 而不是 NPU**: torch 是 2.10.0+cpu 版本，不含 CUDA C++ 扩展。torch_npu 做了 monkey-patch 让 `tensor.to("npu:0")` 在简单场景下工作，但 `nn.Module.to("npu:0")` 内部走 `_apply(convert)` 递归遍历子模块参数时，仍然触发 `torch.cuda._lazy_init()` 并断言 CUDA 编译支持失败。所以 encoder 这类单独加载的小模型（~2GB）放在 CPU 上。

**修复**:

1. `config.py` 不再指定 device，让工厂函数用默认值 None：
```diff
- factory_args={"device": "cuda", "dtype": None}
+ factory_args={"dtype": None}
```

2. `stages.py` 中 create_image_encoder_executor / create_audio_encoder_executor 在 device=None 时自动检测：
```diff
- if device is None:
-     device = get_device_string(0)
+ if device is None:
+     device = "cpu" if get_device_type() == "npu" else get_device_string(0)
```

---

### P2: CUDA_VISIBLE_DEVICES 对 NPU 无效 — TP 进程 GPU 隔离失败

**现象**: `--thinker-tp-size 2 --thinker-gpus 0,1` 启动后，两个 thinker TP 进程同时 OOM 在物理 NPU 0。日志显示 thinker_tp0 和 thinker_tp1 的 `NPU out of memory` 错误都指向同一块物理 NPU，显存分配值完全一致（如 `4.97 GiB already allocated`）。

**分析**: 代码通过设置 `CUDA_VISIBLE_DEVICES` 来限制每个 TP 进程只看到一个 GPU。对 CUDA 设备有效，对 torch_npu 无效。验证结果：
```
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch_npu.device_count()  →  16  （应该是 1）
```
NPU 使用 `ASCEND_RT_VISIBLE_DEVICES` 做设备隔离。

**修复** (`stage_workers.py`): `get_stage_process_env()` 和 `_prepare_cuda_environment()` 中所有硬编码的 `"CUDA_VISIBLE_DEVICES"` 替换为 `get_visible_devices_env()`，该函数在 NPU 上返回 `"ASCEND_RT_VISIBLE_DEVICES"`。

---

### P3: HCCL 8-way 建链死锁

**现象**: TP=8 启动 speech 模式时，8 个 thinker 进程在 `Init torch distributed` 阶段卡死。日志显示 8 个 `Init torch distributed begin`，但只有 1 个 `Init torch distributed ends`，其余 7 个进程永远等不到 HCCL 连接建立，最终超时或被 kill。每次重启后现象随机——有时 2 个通过、有时 1 个。容器反复 crash（exit 137）。

**分析**: HCCL（NPU 等效于 NCCL）在 8 进程同时建链时，多个进程竞争同一 TCP 端口号，导致部分进程端口绑定失败，后续进程一直等待。`HCCL_NPU_SOCKET_PORT_RANGE` 只扩大可用端口范围，无法解决并发抢占的根本问题。`ASCEND_AUTO_CONNECT=0` 禁用 HCCL 自动建链，改为程序显式控制连接顺序。

**修复**: 启动时设置环境变量：
```bash
export ASCEND_AUTO_CONNECT=0
export HCCL_NPU_SOCKET_PORT_RANGE=25000-35000
```
（后续验证 `ASCEND_AUTO_CONNECT=0` 是必要修复，`HCCL_NPU_SOCKET_PORT_RANGE` 是辅助）

---

### P4: avail_gpu_mem() 在 NPU 上返回 None

**现象**: thinker 启动日志中 `pre_load_avail_mem=None`，随后 `RuntimeError: Not enough memory`。同一配置下每次故障时 avail mem 数值不变（始终 ~12GB），不受 `--mem-fraction-static` 参数影响。

**分析**: `utils/misc.py` 的 `avail_gpu_mem()` 函数调用 `torch.cuda.is_available()` 检查 GPU 可用性。在 NPU 上 `torch.cuda.is_available()` 返回 False（torch 2.10.0+cpu 版本），函数直接返回 None。SGLang 收到 None 后使用默认内存计算路径（假设总显存 61GB 全部可用），得出错误的池大小。

**修复** (`misc.py`): 增加 NPU 分支，调用 `torch.npu.mem_get_info(gpu_id)` 获取可用显存。

---

### P5: 内存池公式导致的 mem_fraction 困境

**现象**: `mem_fraction_static=0.05~0.60` 均报 `Not enough memory`，只有 0.75 能通过。

**分析**: SGLang 的 KV cache 池计算公式：
```
available = post_free - pre_free × (1 - mem_fraction)
```
其中 `pre_free` 是 HCCL 初始化后剩余的可用显存。由于 HCCL TP=8 的通信缓冲区占用了每张卡约 47GB，`pre_free` 只有 ~13GB。要使 `available >= 0`：
```
mem_fraction ≥ model_bytes / pre_free = 7.4 / 13 ≈ 0.57
```
池大小 = `mem_fraction × 总显存 = 0.75 × 61 = 45.75GB`。但这是**预留上限**，实际从 `pre_free` 减去模型后再分配，不会真的占 45GB。如果预留量和模型加起来超过 `pre_free`，加载阶段就会 OOM。

**修复**: 使用 `--mem-fraction-static 0.75`。

---

### P6: TTS 请求 preprocessor 格式不兼容

**现象**: 发送 `POST /v1/audio/speech {"input":"你好世界"}` 后返回 500，日志显示 `ValueError: Preprocessing expects a list of chat messages`。

**分析**: `/v1/audio/speech` 端点的 `input` 字段是纯字符串（如 "你好世界"），通过 `GenerateRequest(prompt="你好世界")` 传到底层后，`OmniRequest.inputs` 就是原始字符串。preprocessor 的 `_call_impl()` 对非 dict 类型的 inputs 直接赋值给 `messages = inputs`（即字符串），然后调用 `normalize_messages("你好世界")`，该函数要求输入必须是 list，触发 ValueError。

**修复** (`preprocessor.py`): 对字符串类型 inputs，包装为 chat messages 格式：
```python
if isinstance(inputs, str):
    messages = [{"role": "user", "content": inputs}]
```

---

### P7: Talker 模型初始化时 `_device` 属性缺失

**现象**: 尝试设置 `TORCH_COMPILE_DISABLE=1` 后 talker 启动失败：`'Qwen3OmniTalker' object has no attribute '_device'`。

**分析**: talker 的 `__init__` 中通过 `device = self.model.codec_embedding.weight.device` 获取设备引用，但只存为局部变量。后续 forward 中的 `SamplingBatchInfo(..., device=self._device, ...)` 引用了从未赋值的 `self._device`。正常编译路径下该代码因为 inductor 融合被跳过或内联，不触发此 bug；关闭编译后才暴露。

**修复** (`talker.py`): 在 `__init__` 中添加 `self._device = device`。

---

### P8: 视频解码 torchcodec 依赖 NVIDIA CUDA

**现象**: 发送视频请求后返回 500，日志显示 `OSError: libnvrtc.so.13: cannot open shared object file`。

**分析**: `qwen_vl_utils` 的视频读取后端优先级是 `torchcodec → decord → torchvision`。torchcodec 在 import 时尝试加载 `libnvrtc.so.13`（NVIDIA CUDA Runtime Compiler），NPU 环境没有 NVIDIA CUDA 库。`is_torchcodec_available()` 只检查模块能否导入，不检查运行时依赖。

**修复** (`video.py`): 模块加载时，若检测到 NPU 环境，从 `VIDEO_READER_BACKENDS` 中移除 `"torchcodec"`，让框架自动回退到 decord 或 torchvision（CPU）。

---

### P9 / P10: qwen-vl-utils 版本兼容与返回值变更

**现象**:
- `AttributeError: module has no attribute 'VIDEO_MIN_PIXELS'` / `'IMAGE_FACTOR'`
- `ValueError: too many values to unpack (expected 2)`

**分析**: 新版 qwen-vl-utils API 变更：(1) `VIDEO_MIN_PIXELS` 等属性重命名为 `VIDEO_MIN_TOKEN_NUM`，且从预先计算的像素数变成了 token 数；(2) 视频 reader 返回值从 `(video, fps)` 变为 `(video, metadata, fps)`。旧版 omni 代码用新版库时版本不匹配。

**修复** (`video.py`):
- 用 `hasattr` 检测新版属性名，按 pixel_factor 转换像素数回填旧属性名
- 解包时取 `result[0]`（video）和 `result[-1]`（fps），兼容 2 值和 3 值返回

---

### P11: CPU 视频帧处理 OpenBLAS 线程爆炸

**现象**: 视频请求超时 300s 无返回，日志刷满 OPENBLAS 警告。

**分析**: torchvision 解码视频后在 CPU 上做 resize，OpenBLAS 在 240 核的 ARM 服务器上创建了过多线程，线程上下文切换开销远超实际计算。单帧处理变成不可接受的慢。

**修复**: 启动时限制线程数：`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`。

---

## 6. 待优化点

以下调整均是被迫采取的临时方案，每项都说明了为什么当前无法按标准方式处理、以及后续应如何修复。

---

### O1: ~Encoder 模型降级到 CPU~ ✅ 已解决

**问题**: torch 2.10.0+cpu 版本不支持 `module.to("npu:0")`（C++ 层 `_apply` 触发 `torch.cuda._lazy_init()` 断言失败），导致 encoder 权重无法搬到 NPU。

**解决**: 在 `weight_loader.py` 的 `load_module` 中增加 NPU 路径：
1. 用 `safe_open(device="npu:0")` 将权重从磁盘直接读到 NPU 内存（仿照 SGLang 的 Thinker/Talker 做法）
2. `load_state_dict(assign=True)` 将 NPU 上的参数赋值给模型
3. 逐参数和逐 buffer 用 `_p.data = _p.data.to("npu:0")` 搬运（绕过 `module.to()`）
4. `stages.py` 中 encoder factory 恢复 `get_device_string(0)` 返回 `"npu:0"`

涉及文件: `weight_loader.py`, `stages.py`

---

### O2: 视频解码降级到 CPU

**当前规避手段**: video.py 在模块加载时检测 NPU，从 `qwen_vision.VIDEO_READER_BACKENDS` 中移除 `"torchcodec"`，框架自动回退到 decord 或 torchvision（纯 CPU）。同时设置 `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` 限制 CPU 处理线程数防超时。

**为什么需要临时规避**: `qwen-vl-utils` 的视频读取后端优先级为 torchcodec → decord → torchvision。torchcodec 是 PyTorch 官方视频解码库，底层依赖 FFmpeg + NVIDIA CUDA Runtime（`libnvrtc.so.13`），NPU 环境没有这些 NVIDIA 库。torchcodec 的加载逻辑在 import 阶段就尝试加载动态库，失败即抛 OSError。

**为什么当前无法解决**: Ascend 910B 没有 NVIDIA CUDA 库，也没有等价的 NPU 硬件解码 Python 接口。torchcodec 本身也不支持 NPU 后端。

**后续优化方向**: 方案一：华为发布 torchcodec 的 Ascend 适配版，替换解码后端。方案二：使用 CANN 提供的 DVPP（Digital Vision Pre-Processing）硬件解码 + 自研视频 reader。

---

### O3: `ASCEND_AUTO_CONNECT=0` 禁用 HCCL 自动建链

**当前规避手段**: 启动时设置 `export ASCEND_AUTO_CONNECT=0`，配合 `HCCL_NPU_SOCKET_PORT_RANGE=25000-35000` 扩大可用端口范围。

**为什么需要临时规避**: TP=8 的 8 个 thinker 进程并发初始化 HCCL 分布式组时，HCCL 默认的自动建链模块（auto-connect）会在短时间内从 8 个进程同时发起 TCP 连接请求，竞争同一端口号范围，导致部分进程端口绑定失败（`Communication_Error_Bind_IP_Port`），其余进程一直等待，最终死锁。

**为什么当前无法解决**: 这是 HCCL/CANN 驱动层的并发建链逻辑问题，需华为修复 HCCL 的端口分配算法或在驱动中加入重试机制。

**后续优化方向**: 升级 CANN 后尝试移除 `ASCEND_AUTO_CONNECT=0`。

---

### O4: `mem_fraction_static=0.75` 异常偏高

**当前规避手段**: 使用 `--mem-fraction-static 0.75`，通过 CLI 参数强制设置较高的 mem_fraction 使池公式为非负数。

**为什么需要临时规避**: SGLang 的 KV cache 池计算公式 `available = post_free - pre_free × (1 - mem_fraction)` 中，`pre_free` 是 HCCL 通信缓冲区建立后的可用显存。CANN 9.0.0 的 HCCL 驱动为 TP=8 预留了每卡约 47GB 的通信缓冲，导致 `pre_free` 降至 ~13GB。`mem_fraction` 必须 ≥ 0.57 才能使公式出非负数。标准场景（CUDA）下 NCCL 不会占用如此大量的通信缓冲，`mem_fraction` 通常在 0.3~0.5 之间。

**为什么当前无法解决**: HCCL 通信缓冲大小由 CANN 驱动控制，无法通过环境变量或代码调节。

**后续优化方向**: 升级 CANN 到 HCCL 通信内存优化版本后，将 `mem_fraction_static` 降回正常范围（0.4~0.5），释放更多显存给 KV cache 以提升并发能力。

---

### O5: CANN 编译器不支持 talker 算子组合（CP4 阻塞根因）

**当前规避手段**: 无法规避。现阶段 speech 模式下 talker_ar stage 加载成功后，任何 TTS 请求均会因为首次推理触发 inductor 编译而崩溃。只能使用 text-only 模式（不加载 talker），输出仅限文字。

**为什么需要临时规避**: torch_npu 会在每次 NPU 算子执行时自动尝试 inductor 编译。Thinker 的算子组合（matmul + softmax + layernorm）CANN 编译器能处理，但 Talker 的 Gumbel-Max 采样路径（uint32 cast + argmax + log + gumbel noise）CANN 编译器不支持。由于 torch_npu inductor 是强制自动编译且不可关闭，Talker 第一次推理必然失败。

**为什么当前无法解决**: 所有尝试关闭或绕过 inductor 编译的方法均不生效（TORCHDYNAMO_DISABLE、TORCH_COMPILE_DISABLE、triton ascend backend 替换等均无效，因为 inductor 在 torch_npu 的 transfer_to_npu monkey-patch 层被硬编码注册）。删除 triton backend 会导致 SGLang 模型加载崩溃。这是 CANN 编译器能力缺口，非代码层面可绕过。

**后续优化方向**: 等待华为发布修复该编译器 bug 的新版 CANN（9.1+）。升级后 talker 首次推理的编译应能自动通过，TTS 即可正常工作。

---

### O6: OpenBLAS 线程硬编码

**当前规避手段**: 启动时设置 `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4`。

**为什么需要临时规避**: CPU 视频帧 resize 使用 torchvision 的 `transforms.resize`，底层调用 OpenBLAS。在 240 核 ARM 服务器上，OpenBLAS 默认创建与 CPU 核数相当的线程，线程上下文切换开销远超实际 resize 计算本身，导致单次视频请求超时 300s+。

**为什么当前无法解决**: 最优线程数取决于具体服务器配置，没有普适的默认值。

**后续优化方向**: 在服务启动时自动检测 CPU 核数，按 `max(4, cpu_count // 16)` 动态设置 `OMP_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS`。或当视频解码换用 NPU 硬件时（见 O2），此问题自然消除。

---

### O7: qwen-vl-utils API 手动版本兼容

**当前规避手段**: video.py 中用 `hasattr` 检测新旧属性名，按 pixel_factor 换算后回填缺失属性。视频 reader 返回值用 `result[0], result[-1]` 兼容 2 值和 3 值返回。

**为什么需要临时规避**: 远程 Docker 镜像中预装的 `qwen-vl-utils` 版本与 sglang-omni 开发时使用的版本不一致，部分属性名变更（`VIDEO_MIN_PIXELS` → `VIDEO_MIN_TOKEN_NUM`，`IMAGE_FACTOR` 被移除），视频 reader 返回值数量从 2 变为 3。容器内无法固定依赖版本（镜像预装，非 pip 安装）。

**为什么当前无法解决**: 需要在 Docker 镜像构建时固定 `qwen-vl-utils` 版本或要求 omni 声明精确依赖。

**后续优化方向**: 在 `pyproject.toml` 中固定 `qwen-vl-utils` 版本，Docker 镜像中安装匹配版本后，移除 `video.py` 中的 `hasattr` 兼容代码。

---

## 7. 最终启动配置

### Text-only 模式（CP1-CP7）

```bash
export ASCEND_AUTO_CONNECT=0
export HCCL_NPU_SOCKET_PORT_RANGE=12000-22000
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

sgl-omni serve \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --text-only \
  --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 \
  --mem-fraction-static 0.75
```

### Speech 模式（CP1-CP3 含 talker，CP4 不可用）

```bash
export ASCEND_AUTO_CONNECT=0
export HCCL_NPU_SOCKET_PORT_RANGE=12000-22000

sgl-omni serve \
  --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 \
  --talker-gpu 8 \
  --mem-fraction-static 0.75
```

---

## 8. 更新记录

- 2026-07-10: 环境检查、代码同步
- 2026-07-10: config.py / stages.py / stage_workers.py / misc.py 修复
- 2026-07-10: ✅ text-only TP=8 端到端文本+图片验证
- 2026-07-10: ASCEND_AUTO_CONNECT=0 修复 HCCL 死锁
- 2026-07-10: ✅ speech 模式 9 Stage 启动成功
- 2026-07-10: preprocessor.py / talker.py 修复
- 2026-07-14: ✅ CP5 音频输入 / CP7 多模态混合验证
- 2026-07-14: video.py NPU torchcodec skip + qwen-vl-utils 兼容
- 2026-07-15: ✅ CP6 视频输入验证
- 2026-07-15: CP4 深度调试 — sampler float64→float32、distributed try/except、triton.compile interceptor、_precompile_worker monkey-patch
- 2026-07-15: ❌ CP4 最终卡在 CANN 9.0.0 BiSheng LLVM 指令选择 bug，需 CANN 升级
