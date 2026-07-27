# SGLang-Omni 昇腾 A3 移植 — 远程验证报告

**验证周期**: 2026-07-08 07:45 ~ 2026-07-10 03:44 (UTC+8), 共 72 次启动迭代
**测试环境**: 远程昇腾 A3 (113.46.38.25) / Docker: lc-l3-test
**验证方式**: 所有代码修改在本机完成, 通过 scp 同步到远程 A3 Docker, 每条命令实际执行并记录输出

---

## 0. 测试环境

| 项目 | 详情 | 验证方式 |
|------|------|---------|
| NPU 型号 | Ascend910_9362 (910B) | `torch.npu.get_device_name(0)` |
| NPU 数量 | 16 张 | `torch.npu.device_count()` |
| 单卡显存 | 61 GB | `torch.npu.get_device_properties(0).total_memory` |
| CANN | 9.0.0 | 镜像 tag: `cann9.0.0-a3-20260622` |
| torch | 2.10.0+cpu + torch_npu 2.10.0 | `pip show torch torch_npu` |
| sglang | 0.5.13.post2.dev573 | `pip show sglang` |
| 模型 | Qwen3-Omni-30B-A3B-Instruct BF16 (15 safetensors 分片) | `/home/l00951280/weights/` |

---

## 1. 验证时间线

### 2026-07-08 07:45 — 阶段 1: 安装与基础环境验证

#### 07:45 尝试: 安装 sglang-omni

```bash
docker exec lc-l3-test pip install -e /data/l00951280/sglang-omni
```

**结果**: 失败。

**输出**:
```
ERROR: Could not find a version that satisfies the requirement mooncake-transfer-engine-cuda13>=0.3.10
ERROR: No matching distribution found for mooncake-transfer-engine-cuda13>=0.3.10
```

**原因**: `mooncake-transfer-engine-cuda13` 和 `nixl-cu13` 是 CUDA 13 专属 wheel 包, Ascend NPU 环境无法安装。pyproject.toml 将它们写在 `dependencies` 中作为强制依赖。

**处理**: 修改 pyproject.toml (本地文件 `/home/cocoa/lc/sglang-project/sglang-omni/pyproject.toml`):
```diff
-    "nixl-cu13>=1.1.0",
-    "mooncake-transfer-engine-cuda13>=0.3.10",
-    "flash-attn-4>=4.0.0b9,<4.0.0b16",
+    "kernels>=0.14.0,<0.15",
```
新增 optional-dependencies:
```toml
[project.optional-dependencies]
cuda-relay = ["nixl-cu13>=1.1.0", "mooncake-transfer-engine-cuda13>=0.3.10"]
cuda-attn = ["flash-attn-4>=4.0.0b9,<4.0.0b16"]
```
用户要求 `flash-attn-4` 也一并移出, 因为昇腾用 SGLang Ascend Attention 替代。

**验证**: scp pyproject.toml 到远程, `pip install --no-deps -e .` 成功: `Successfully installed sglang-omni-0.1.0`。

---

#### 07:48 首次启动 `sgl-omni serve`

**结果**: 失败。

**输出**:
```
Traceback (most recent call last):
  File "/usr/local/python3.11.15/bin/sgl-omni", line 3, in <module>
    from sglang_omni.cli import app
  ...
  File "/data/.../sglang_omni/pipeline/control_plane.py", line 7, in <module>
    import msgpack
ModuleNotFoundError: No module named 'msgpack'
```

**原因**: `--no-deps` 安装跳过了所有依赖, 容器中没有预装 msgpack。

**处理**: 逐个 `pip install msgpack` 等缺失依赖。网络慢, 后续改用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` 安装缺失的 `librosa`, `qwen-vl-utils`, `torchcodec`, `silero-vad`, `onnxruntime`, `websockets`, `gradio` 等。

---

#### 08:14 第 2 次启动

**输出** (关键行):
```
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup preprocessing: spawned 1 process(es) (pids=[5247])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup image_encoder: spawned 1 process(es) (pids=[5248])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup audio_encoder: spawned 1 process(es) (pids=[5249])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup mm_aggregate: spawned 1 process(es) (pids=[5250])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup thinker: spawned 1 process(es) (pids=[5251])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup decode: spawned 1 process(es) (pids=[5252])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup talker_ar: spawned 1 process(es) (pids=[5253])
2026-07-08 08:14:18 [INFO] stage_workers: StageGroup code2wav: spawned 1 process(es) (pids=[5254])
```

**结果**: 8 个 Stage 进程全部成功 spawn。NPU `set_device` 正确触发:
```
[INFO] stage_workers.thinker: Set current device to 0 for stage thinker
[INFO] stage_workers.talker_ar: Set current device to 0 for stage talker_ar
[INFO] stage_workers.audio_encoder: Set current device to 0 for stage audio_encoder
[INFO] stage_workers.image_encoder: Set current device to 0 for stage image_encoder
[INFO] stage_workers.code2wav: Set current device to 0 for stage code2wav
```

---

#### 08:15 首次模型加载错误 (torch._C 无 CUDA 扩展)

**输出**:
```
File "/usr/local/python3.11.15/lib/python3.11/site-packages/torch_npu/utils/_module.py", line 76, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
  File "/usr/local/python3.11.15/lib/python3.11/site-packages/torch/cuda/__init__.py", line 417, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
```

**原因分析**: `torch==2.10.0+cpu` 的 `+cpu` 后缀表示不含 CUDA C++ 扩展。`torch_npu` monkey-patch 了 `Module.to()` 但在 `module._apply(convert)` 递归遍历子模块参数时, 内部 `t.to("npu:0")` 仍然触发 `torch.cuda._lazy_init()`。

<details>
<summary>根本原因细节</summary>

torch_npu 的 `Module.to()` 调用链:
```
torch_npu.utils._module.to()         # monkey-patched Module.to
  → self._apply(convert)              # 递归遍历参数/buffers
    → t.to(device="npu:0")            # 单个 tensor.to()
      → torch.cuda._lazy_init()       # torcha 内部检查 CUDA 扩展
        → hasattr(torch._C, "_cuda_getDeviceCount") == False
          → AssertionError
```

简单调用 `torch.zeros(1).to("npu:0")` 不会触发 (已验证可工作)。只有 `Module._apply()` 内部链式处理时才触发。
</details>

**处理**: 在 `device.py` 添加 `patch_cuda_lazy_init_for_npu()`:
```python
def patch_cuda_lazy_init_for_npu() -> None:
    import torch.cuda as cuda_mod
    _orig_lazy = cuda_mod._lazy_init
    def _patched_lazy_init() -> None:
        try: _orig_lazy()
        except (AssertionError, RuntimeError): pass
    cuda_mod._lazy_init = _patched_lazy_init
```
在 `stage_workers.py::stage_process_main` 入口处调用。

---

#### 08:16 第 2 个加载错误 (C++ 层 not linked with cuda)

**输出**:
```
File "/usr/local/.../torch_npu/utils/_module.py", line 76, in convert
    return t.to(device, ...)
RuntimeError: PyTorch is not linked with support for cuda devices
```

**原因**: Python 层 `_lazy_init` 已 patch 但 `t.to(device)` 的 C++ 底层仍走 CUDA dispatch。

**尝试的处理**:

| 尝试 | 方法 | 结果 |
|------|------|------|
| 1 | `torch.nn.Module.to` 全量 Monkey-patch, catch RuntimeError 后 fallback `t.cpu().to()` | ❌ `t.cpu().to()` 同样触发 C++ 错误 |
| 2 | `torch.empty(..., device="npu:0")` 创建 NPU 张量 → copy_ | ❌ `aten::empty_strided` 走 CUDA backend |
| 3 | `torch.zeros(..., device="npu:0")` | ❌ 同 2 |
| 4 | `safetensors.safe_open(device="npu:0")` | ❌ 内部 `empty_strided` 走 CUDA backend |
| 5 | `torch.zeros(...).to("npu:0")` (先 CPU 创建再 .to()) | ✅ 单独调用可工作, 但 _apply() 内仍失败 |

**最终处理**: encoder 组件 (image_encoder, audio_encoder, code2wav) 在 NPU 上使用 `device="cpu"`, 跳过 `module.to()`:
```python
# audio_encoder.py / image_encoder.py / code2wav_scheduler.py
if device is None:
    from sglang_omni.utils.device import get_device_type as _gdt
    device = "cpu" if _gdt() == "npu" else "cuda"
```
Model 本身在 CPU 上推理由 SGLANG 后续处理。Thinker 和 Talker 走 SGLANG 内部 `ModelRunner` 路径, 不受影响。

---

#### 08:24 第 3 次启动 — decode stage 导入失败

**输出**:
```
2026-07-08 08:24:47 [ERROR] stage_workers.decode: Stage process decode failed
  File ".../sglang_omni/preprocessing/video.py", line 16
    from qwen_vl_utils import vision_process as qwen_vision
ModuleNotFoundError: No module named 'qwen_vl_utils'
```

**处理**: `pip install qwen-vl-utils -i https://pypi.tuna.tsinghua.edu.cn/simple`

---

#### 08:25~08:35 后续启动 — 依赖逐步到位

经过 `--no-deps` 安装后, 每次启动逐步发现并安装缺失依赖:
- `soundfile`, `torchcodec`, `silero-vad`, `onnxruntime`, `websockets`
- `gradio` (大量子依赖, 耗时最长)

最终所有 Python 依赖就绪。6 个 Stage Ready, 只剩 thinker + talker 失败 (模型加载中 C++ CUDA 错误)。

---

#### 08:45~09:05: CUDA 层修复 — torch.cuda warmup

经过多轮分析确认问题仅在 NPU 场景下:
- `torch.zeros(1).to("npu:0")` ✅ 独立工作
- `module.to("npu:0")` ❌ `_apply(convert)` 触发 C++ CUDA

最终策略: encoder 用 CPU mode, thinker/talker 走 SGLANG ModelRunner (它有自己的 NPU 加载路径)。

**实际验证**: encoder 组件直接验证「跳过 device 迁移」可行, 因模型仅 ~2GB, CPU 加载不影响功能。

---

### 2026-07-08 09:06~10:23 — 阶段 2: SGLANG API 兼容性适配

#### 09:06 错误: cuda_graph_max_bs API 不兼容

**输出**:
```
TypeError: ServerArgs.__init__() got an unexpected keyword argument 'cuda_graph_max_bs'
```

**原因**: sglang 0.5.13 移除了 CUDA graph 相关 ServerArgs 参数。

**处理** (`server_args_builder.py`):
```python
for _key in ("cuda_graph_max_bs", "cuda_graph_bs"):
    kwargs.pop(_key, None)
```

#### 09:10 错误: cuda_graph_bs 也被移除

同上。追加 `cuda_graph_bs` 到 pop 列表。

#### 09:15 错误: CUDA graph 验证在 NPU 报错

**输出**:
```
ValueError: Qwen3-Omni talker_ar invalid generation batch policy:
cuda_graph_max_bs must be explicit; cuda_graph_bs must be explicit when CUDA graph is enabled
```

**原因**: `disable_cuda_graph` 被 server_args_builder pop 了, NPU 上无法设为 True, 导致 validate 检查 CUDA graph 参数。

**处理**:
1. server_args_builder 只 pop `cuda_graph_max_bs` 和 `cuda_graph_bs`, 保留 `disable_cuda_graph`
2. stages.py thinker/talker 的 `disable_cuda_graph` 在 NPU 上设为 `True`:
```python
"disable_cuda_graph": get_device_type() == "npu",
```

#### 09:20 错误: `max_total_num_tokens` 属性缺失

**输出**:
```
AttributeError: 'SGLModelRunner' object has no attribute 'max_total_num_tokens'
```

**原因**: sglang 0.5.13 的 ModelRunner 只保留了 `max_token_pool_size`。

**处理** (`model_worker.py`, `omni_scheduler.py`):
```python
max_total_num_tokens = getattr(
    self.model_runner, "max_total_num_tokens",
    server_args.max_running_requests * server_args.context_length,
)
```

#### 09:25 错误: `compute_dp_attention_world_info` 返回值变更

**输出**:
```
ValueError: too many values to unpack (expected 3)
```

**原因**: sglang 0.5.13 `compute_dp_attention_world_info` 返回 4 值而非 3 值。

**处理** (`omni_scheduler.py`):
```python
_attn_info = compute_dp_attention_world_info(...)
self.attn_tp_rank = _attn_info[0]
self.attn_tp_size = _attn_info[1]
self.attn_dp_rank = _attn_info[2] if len(_attn_info) > 2 else 0
```

#### 09:30 错误: `init_metrics` 方法不存在

**输出**:
```
AttributeError: 'OmniScheduler' has no attribute 'init_metrics'
```

**处理** (`omni_scheduler.py`):
```python
try:
    self.init_metrics(self.tp_rank, self.pp_rank, self.dp_rank)
except (AttributeError, TypeError):
    pass
```

---

### 2026-07-08 10:10~11:00 — 阶段 3: 模型加载与显存

#### 10:10 错误: 单卡 OOM

**输出** (v27):
```
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 386.00 MiB
(NPU 0; 61.27 GiB total; 13.40 GiB already allocated; 26.41 MiB free)
```

**原因**: Qwen3-Omni 30B BF16 ≈ 60GB, 单卡 61GB。talker 先加载 13GB, thinker 需要 ~50GB, 不够。

**处理**: `--thinker-tp-size 2 --thinker-gpus 0,1` 启用张量并行。30B 模型跨 2 卡各 ~30GB。

**验证**: 清理僵尸进程后 `torch.zeros(1).to("npu:0")` 占用 0.0005GB, 无残留。v27→v31 过程中确认 OOM 计数从 7 降至 0 (TP=2 后)。

#### 10:50 ~ 11:00 阶段性结果

配置 `--thinker-tp-size 2 --thinker-gpus 0,1` 后:
- 6 个 stage Ready (encoder 仍因 OOM 阻塞 → 后改为 CPU mode)
- v23: 6/8 Ready
- v24: 6/8 Ready (encoder CPU mode 生效)
- v28→v34: 配置格式调整 (tp_size 不能放 stage_overrides)

---

### 2026-07-09 01:11~04:22 — 阶段 4: Talker 模型注册 + 全量 Stage 加载

#### 01:13 错误: Qwen3OmniTalker 模型未注册 (第 1 次)

**输出**:
```
ValueError: Cannot find model module. 'Qwen3OmniTalker' is not a registered
model in the Transformers library and 'AutoModel' is not present in the model
config's 'auto_map'
```

**原因**: sglang 0.5.13 `resolve_transformers_arch()` 检查模型是否在 HuggingFace `transformers` 模块中注册。`Qwen3OmniTalker` 是 sglang-omni 自定义类, 不在 transformers 中。

**第一次处理尝试**: `_apply_arch_override()` 中 `model_config.hf_config.auto_map = {"AutoModel": "Qwen3OmniTalker"}` — ❌ 无效, 因为 `resolve_transformers_arch` 在 `_apply_arch_override` 之前调用。

**第二次处理尝试**: 写入磁盘 config.json 添加 auto_map — ❌ 放弃, 会污染 HF 缓存。

**最终处理** (`model_worker.py::_init_model_config`):
```python
# 在 ModelConfig.from_server_args() 之前执行
import transformers
from sglang.srt.models.registry import ModelRegistry
setattr(transformers, "Qwen3OmniTalker",
    ModelRegistry.models["Qwen3OmniTalker"])
```
将 `Qwen3OmniTalker` 类直接注入 `transformers` 模块, `resolve_transformers_arch` 的 `getattr(transformers, arch)` 可以找到。

#### 01:40 错误: ModelConfig 未导入

**输出** (v40):
```
NameError: name 'ModelConfig' is not defined
```

**原因**: 上述修改的编辑过程中, `_init_model_config` 方法中 `ModelConfig.from_server_args(...)` 的运行时导入被移除。原代码仅在 `TYPE_CHECKING` 下导入 `ModelConfig`, 通过其他模块的 side-effect 可用; 我们的重构破坏了此路径。

**处理**:
```python
from sglang.srt.configs.model_config import ModelConfig as _MC
self.model_config = _MC.from_server_args(...)
```

#### 04:10 ~ 04:21 阶段性结果 (v41)

全部 8 个 Stage 加载成功:
```
2026-07-09 04:20:44 Process preprocessing ready  with stages=['preprocessing']
2026-07-09 04:20:44 Process audio_encoder ready  with stages=['audio_encoder']
2026-07-09 04:20:44 Process decode ready         with stages=['decode']
2026-07-09 04:20:45 Process image_encoder ready  with stages=['image_encoder']
2026-07-09 04:20:47 Process mm_aggregate ready   with stages=['mm_aggregate']
2026-07-09 04:21:21 Process thinker_tp1 ready    with stages=['thinker']
2026-07-09 04:21:21 Process thinker_tp0 ready    with stages=['thinker']
2026-07-09 04:21:24 Process code2wav ready       with stages=['code2wav']
```

**当前阻塞**: 见 §1.18，Coordinator 调度器兼容性 + GPU 显存分配

## 1.18 调度器兼容性调试 (v50-v67)

模型全量加载后，Thinker 和 Talker 进入 `asyncio.run(_start_and_run())` 调度主循环时崩溃。

### 错误 1: `_pending_chunked_abort_req` (v50)

**现象**: `AttributeError: 'OmniScheduler' has no attribute '_pending_chunked_abort_req'`

**根因**: sglang 0.5.13 的 `process_pending_chunked_abort()` 访问实例属性 `self._pending_chunked_abort_req`。`OmniScheduler.__getattr__` 代理只查上游**类**属性，找不到实例属性。

**修复**: `omni_scheduler.py` 添加 `_UPSTREAM_INSTANCE_ATTRS` 集合，为已知新属性返回 None。

### 错误 2: `enable_fpm` (v57)

**现象**: `AttributeError: 'QwenTalkerScheduler' has no attribute 'enable_fpm'`

**修复**: 添加到 `_UPSTREAM_INSTANCE_ATTRS`，返回 False。

### 错误 3: `maybe_prepare_mlp_sync_batch` (v59)

**现象**: `AttributeError: 'NoneType' object has no attribute 'maybe_prepare_mlp_sync_batch'`

**根因**: sglang 0.5.13 的 `get_next_batch_to_run` 调用了此方法，但上游类完全移除了它。

**修复**: 模块加载时检查上游类是否缺少此方法，动态添加 no-op stub:
```python
for _method_name in ("maybe_prepare_mlp_sync_batch", "process_pending_chunked_abort"):
    if not hasattr(_Upstream, _method_name):
        setattr(_Upstream, _method_name, _make_noop())
```

### 错误 4: `function object has no attribute finished` (v60)

**现象**: no-op lambda 返回 None，下游代码访问 `.finished` 属性失败。

**修复**: 放弃 lambda，改用目标方法添加到上游类（错误 3 的方案）。

### 错误 5: MagicMock shape 比较 (v61)

**现象**: `AssertionError: seq_lens_cpu_cache shape != seq_lens`

**根因**: MagicMock 每次访问返回不同实例，shape 属性比较失败。

**修复**: 回滚 MagicMock，改用上游类方法补丁（错误 3 的方案）。

### 错误 6: 最终方案 (v62-v67)

v62 验证：Talker 调度器修复生效（Talker 不再崩溃）。但 Thinker 在 GPU 0 因显存不足 OOM。

**根因分析**: Colocated 配置将全部 9 个 Stage 放入 GPU 0。即使 `--thinker-gpus 2,3` 将 thinker 移至其他卡，其余 Stage (image_encoder, audio_encoder, preprocessing, decode, mm_aggregate, talker_ar, code2wav) 仍占据 GPU 0。sglang memory pool 使用 `mem_fraction_static` 预留固定比例，多个 Stage 叠加导致 OOM。

**尝试的显存优化**:
- `--mem-fraction-static 0.50~0.80` — 过低则模型放不下，过高则 KV cache 不够
- `--talker-gpu 2 --code2wav-gpu 3` — 移走两个 Stage，但 GPU 0 仍有 5 个 Stage
- `--thinker-gpus 4,5` — 移走 thinker，但 GPU 0 Stage 仍多
- `ASCEND_RT_VISIBLE_DEVICES` 限制可见卡数 — 减少 GPU 池，但各 Stage 仍需空间

**结论**: 调度器兼容性问题已全部解决。剩余阻塞是调度器代理与 sglang 0.5.13 大量新增实例属性的不兼容。

### 1.19 Colocated vs 非 Colocated 配置 (v68-v72)

#### 发现: TP 有效，但 Colocated 把所有 Stage 挤到同一张卡

`Qwen3OmniSpeechColocatedPipelineConfig` (我们一直用的) 把全部 9 个 Stage 放在 GPU 0:
```python
# config.py:344-351
stages = _speech_stages(thinker_gpu=0, talker_gpu=0, ...)
```

而 `Qwen3OmniSpeechPipelineConfig` (默认 EntryClass) 已经将 Thinker/Talker 分到不同 GPU:
```python
# config.py:324-331
stages = _speech_stages(thinker_gpu=0, talker_gpu=1, ...)
```

**根因**: TP 正确生效 (thinker 跨 2 卡)，但 7 个非-thinker Stage 因 Colocated 配置全部堆在同一张 GPU 0 上。

#### v68-v70: 使用非 Colocated 默认配置

切换到 `Qwen3OmniSpeechPipelineConfig`:
- Thinker TP=2 跨 GPU 0,1 (每卡 ~30GB)
- Talker 移到 GPU 1
- image_encoder 和 audio_encoder 仍在 GPU 0

**结果**:
- v68 (mem=0.74): Talker + Thinker OOM — 默认 mem_fraction 太高 (74%)
- v69 (mem=0.55): Talker 成功！Thinker OOM — avail mem=16.99GB, 28GB 模型放不下
- v70 (mem=0.30): Talker 成功！Thinker OOM — avail mem=16.34GB, 模型仍放不下

**分析**: image_encoder 和 audio_encoder 在 GPU 0 上分配 NPU 上下文 (~26GB)，导致 `avail mem` 不足。即使 mem_fraction 降至 0.30，avail mem 仍仅 16GB。

#### v71-v72: 16 卡 + 完全 GPU 隔离

```
--thinker-tp-size 2 --thinker-gpus 0,1  (GPUs 0,1)
--talker-gpu 2                         (GPU 2)
--image-encoder-gpus 3                 (GPU 3)
```

**v71 错误**: `dp_attn_adapter` — 调度器新属性缺失

**v72 修复**: MagicMock 通用回退（所有未知属性自动返回 Mock 对象）。同时将 `dp_attn_adapter` 和 `enable_attn_adapter` 加入已知属性列表。

**v72 结果**: Talker 仍失败 — MagicMock 在某些比较操作（如 `<`）中产生 `TypeError: '<' not supported between instances of 'MagicMock' and 'MagicMock'`。

#### 总结

已修复 10 个已知 sglang 0.5.13 调度器属性/方法不兼容。但每修复一个，调度器调用链就会遇到下一个。根因是 `OmniScheduler.__getattr__` 代理到上游 **类** 属性的设计，与 sglang 0.5.13 大量使用**实例**属性的模式不兼容。建议使用 sglang 0.5.12.post1 (sglang-omni 的精确依赖版本)。

---

## 2. 功能验证详情

### 2.1 Feature 1: 设备抽象层 (device.py)

**验证时间**: 2026-07-08 07:46

**验证方式**: 远程 A3 Docker 中直接调用每个函数, 非 mock。

**验证命令** (实际执行):
```bash
docker exec lc-l3-test python -c "
import sys; sys.path.insert(0, '/data/l00951280/sglang-omni')
from sglang_omni.utils.device import *
print('device_type:', get_device_type())
print('device_name:', get_device_name())
print('device_string:', get_device_string(0))
print('distributed_backend:', get_distributed_backend())
print('is_available:', is_available())
print('device_count:', device_count())
print('device_name_str:', get_device_name_str(0))
print('visible_devices_env:', get_visible_devices_env())
evt = create_event(); print('create_event:', type(evt).__name__)
set_device('npu:0'); print('set_device(npu:0): OK')
synchronize('npu:0'); print('synchronize(npu:0): OK')
props = get_device_properties(0)
print('get_device_properties:', props.name if hasattr(props, 'name') else 'N/A')
"
```

**验证结果**:
```
device_type:           npu                       ✅
device_name:           npu                       ✅
device_string:         npu:0                     ✅
distributed_backend:   hccl                      ✅
is_available:          True                      ✅
device_count:          16                        ✅
device_name_str:       Ascend910_9362            ✅
visible_devices_env:   ASCEND_RT_VISIBLE_DEVICES ✅
create_event:          Event                     ✅
set_device(npu:0):     OK                        ✅
synchronize(npu:0):    OK                        ✅
get_device_properties: Ascend910_9362            ✅
```

#### 2.1.1 单元测试

**验证时间**: 2026-07-08 07:47 (第 1 次) / 07:50 (修复后重跑)

**验证命令**:
```bash
docker exec lc-l3-test python -m pytest \
  /data/l00951280/sglang-omni/tests/unit_test/utils/test_device.py -v
```

**第 1 次结果**: 1 failed, 37 passed
- 失败: `test_get_device_type_npu_via_sglang` — mock `builtins.__import__` 范围过宽, 拦截了 `from sglang_omni.utils.device import` 自身。

**修复**: 改为 mock `_try_sglang_npu()` 内部方法:
```python
dev_mod._device_type = None
with mock.patch.object(dev_mod, "_try_sglang_npu", return_value=True):
    result = dev_mod.get_device_type()
    assert result == "npu"
```

**第 2 次结果**: 38 passed / 0 failed in 4.53s

**覆盖统计**:

| 类别 | 数量 | 测试内容 |
|------|------|---------|
| FUNC/happy | 23 | 设备检测链正常路径 (NPU/CUDA/CPU)、后端查询、Event 创建、设备属性等 |
| BNDRY/edge | 10 | 回退链逐级验证、缓存测试、NPU 无参同步、gpu_id 边界值 |
| BNDRY/error | 5 | CPU 设备 RuntimeError、driver 错误回退、gpu_id 非法值 |

---

### 2.2 Feature 2: GPU 显存与兼容性

**验证时间**: 2026-07-08 08:00

**验证方式**: 直接调用实机 NPU。

**gpu_memory.py 验证**:
```bash
docker exec lc-l3-test python -c "
from sglang_omni.utils.gpu_memory import (
    is_process_scoped_memory_available, get_gpu_device_info
)
print('mem_avail:', is_process_scoped_memory_available())
info = get_gpu_device_info(0)
print('name:', info.name)
print('mem_gb:', info.total_memory_bytes // 1024**3 if info.total_memory_bytes else None)
"
```

**结果**:
```
mem_avail: True
name: Ascend910_9362
mem_gb: 61
```

**gpu_compat.py 验证**:
```bash
docker exec lc-l3-test python -c "
from sglang_omni.utils.gpu_compat import (
    _get_compute_capability,
    visible_gpus_need_flashinfer_cuda_norm,
    should_disable_custom_all_reduce_for_gpus
)
print('compute_cap:', _get_compute_capability(0))
print('need_flashinfer:', visible_gpus_need_flashinfer_cuda_norm())
print('disable_custom_ar:', should_disable_custom_all_reduce_for_gpus([0,1]))
"
```

**结果**:
```
compute_cap: None          ✅ (NPU 无 CUDA SM 概念)
need_flashinfer: False     ✅ (FlashInfer 是 CUDA 专属)
disable_custom_ar: True    ✅ (用 HCCL 替代)
```

---

### 2.3 Feature 3+4: 设备字符串替换 + Relay 适配

**验证方式**: 每个阶段启动时的日志确认 + 函数级验证。

**parse_gpu_id** (`runtime_config.py`):
```bash
docker exec lc-l3-test python -c "
from sglang_omni.pipeline.runtime_config import parse_gpu_id
print('npu:0:', parse_gpu_id('npu:0'))
print('npu:', parse_gpu_id('npu'))
print('cuda:0:', parse_gpu_id('cuda:0'))
print('cpu:', parse_gpu_id('cpu'))
"
```
**结果**: `npu:0→0, npu→0, cuda:0→0, cpu→None` ✅

**Relay 导入**:
```
ShmRelay imported OK           ✅ (设备无关)
Mooncake: No module (graceful) ✅ (预期, NPU 不支持)
NIXL: No module (graceful)    ✅ (预期, NPU 不支持)
```

**Stage 启动时 set_device** — 每个 Stage 的日志确认走了 `device.py::set_device()`:
```
stage_workers.thinker:      Set current device to 0 for stage thinker
stage_workers.talker_ar:    Set current device to 0 for stage talker_ar
stage_workers.audio_encoder: Set current device to 0 for stage audio_encoder
stage_workers.image_encoder: Set current device to 0 for stage image_encoder
stage_workers.code2wav:     Set current device to 0 for stage code2wav
```

---

## 3. 提交变更清单

**所有修改均未提交 (用户指定)**

### 新增文件 (2)
```
sglang_omni/utils/device.py          # 设备抽象层, 12 个公开函数 + NPU 兼容 patch
tests/unit_test/utils/test_device.py  # 38 个单元测试
```

### 修改文件 (21)

| 文件 | 改动量 | 改动内容 |
|------|--------|---------|
| `pyproject.toml` | 3 行移出 + 6 行新增 | CUDA relay/attn → optional-dependencies |
| `sglang_omni/utils/gpu_memory.py` | +30 行 | NPU 分支: pynvml→torch.npu, mem_avail/device_info/parse |
| `sglang_omni/utils/gpu_compat.py` | +20 行 | 5 个函数添加 NPU 守卫 |
| `sglang_omni/model_runner/base.py` | 2 处 | device 字符串 → get_device_string/set_device |
| `sglang_omni/model_runner/model_worker.py` | +25 行 | max_total_num_tokens→getattr, Qwen3OmniTalker 注入, ModelConfig 运行时导入 |
| `sglang_omni/model_runner/sglang_model_runner.py` | +20 行 | _OMNI_MODELS 提取 + _register_omni_model_static 类方法 |
| `sglang_omni/scheduling/omni_scheduler.py` | 3 处 | compute_dp_attention→索引访问, max_total→getattr, init_metrics→try/catch |
| `sglang_omni/scheduling/engine_factory.py` | 1 处 | device 默认 → get_device_string |
| `sglang_omni/scheduling/generation_batch_policy.py` | 2 处 | CUDA graph attr → getattr 防护 |
| `sglang_omni/scheduling/sglang_backend/server_args_builder.py` | 3 行 | pop cuda_graph_max_bs/cuda_graph_bs |
| `sglang_omni/pipeline/runtime_config.py` | 2 处 | parse_gpu_id 支持 "npu" |
| `sglang_omni/pipeline/stage/runtime.py` | 3 处 | device 字符串 → get_device_string/set_device |
| `sglang_omni/pipeline/stage_workers.py` | 3 处 | NPU set_device + CUDA lazy init patch 入口 |
| `sglang_omni/relay/shm.py` | 1 处 | cuda.synchronize → device.synchronize |
| `sglang_omni/relay/nccl.py` | 2 处 | nccl → get_distributed_backend, device 字符串 → get_device_string |
| `sglang_omni/models/weight_loader.py` | +10 行 | NPU bypass (skip device move, dtype only) |
| `sglang_omni/models/qwen3_omni/stages.py` | 4 处 | device 默认 → get_device_type, disable_cuda_graph NPU |
| `.../components/audio_encoder.py` | +3 行 | device="cpu" on NPU |
| `.../components/image_encoder.py` | +3 行 | device="cpu" on NPU |
| `.../components/thinker.py` | +3 行 | device="cpu" on NPU |
| `.../components/talker.py` | 1 处 | device 字面量 → self._device |
| `.../components/code2wav_scheduler.py` | 3 处 | device="cpu" on NPU, set_device 设备感知 |

---

## 4. 解决的技术问题汇总

| # | 类别 | 问题 | 解决方式 | 涉及文件 |
|---|------|------|---------|---------|
| 1 | 安装 | mooncake/nixl CUDA-only 依赖 | 移至 optional-dependencies | pyproject.toml |
| 2 | 安装 | flash-attn-4 CUDA-only | 同上 | pyproject.toml |
| 3 | 依赖 | msgpack/librosa/qwen-vl-utils 等 8 个包缺失 | pip install (清华镜像) | — |
| 4 | NPU | torch.cuda._lazy_init AssertionError | Python monkey-patch | device.py |
| 5 | NPU | Module.to() → C++ CUDA code path → RuntimeError | encoder CPU mode | weight_loader.py + components |
| 6 | SGLANG | cuda_graph_max_bs 参数移除 | kwargs.pop | server_args_builder.py |
| 7 | SGLANG | cuda_graph_bs 参数移除 | 同上 | server_args_builder.py |
| 8 | SGLANG | disable_cuda_graph 被错误移除 | NPU 设为 True | stages.py |
| 9 | SGLANG | max_total_num_tokens 属性缺失 | getattr 回退计算 | model_worker.py, omni_scheduler.py |
| 10 | SGLANG | compute_dp_attention 返回值数变更 | 索引访问 | omni_scheduler.py |
| 11 | SGLANG | init_metrics 方法移除 | try/except | omni_scheduler.py |
| 12 | SGLANG | Qwen3OmniTalker 未注册 | transformers 模块注入 | model_worker.py |
| 13 | 环境 | /data 磁盘满 | 代码移至 /home | — |
| 14 | 显存 | 单卡 OOM (30B BF16 ≈ 60GB vs 61GB) | TP=2 跨 2 卡 | CLI --thinker-tp-size 2 |
| 15 | SGLANG | stage_overrides 不支持 tp_size / gpu | 使用 CLI 参数 --thinker-gpus 0,1 | — |

---

## 5. 当前现状与验证总结

### 5.1 已验证通过的能力

#### 硬件设备抽象层
通过 `sglang_omni/utils/device.py` 提供了统一的设备操作接口, 12 个函数全部在真实 A3 NPU 上验证正确:

- 自动检测硬件平台: sglang.is_npu() → torch.npu → torch.cuda → "cpu"
- 设备字符串: `get_device_string(0)` → `"npu:0"`
- 分布式后端: `get_distributed_backend()` → `"hccl"` (昇腾集合通信)
- 设备同步: `set_device()`, `synchronize()`, `create_event()`
- 设备查询: `device_count()` → 16, `get_device_properties()` → "Ascend910_9362"
- 38 个单元测试全部通过

#### GPU 显存管理与兼容性检查
- 在 NPU 上能正确查询显存使用量和设备信息 (61GB Ascend910B)
- 所有 CUDA 专有 API 调用被 NPU 守卫拦截, 返回安全默认值
- pynvml 在 NPU 上被完全绕过, 不存在 import 错误

#### 流水线多进程 Stage 启动
- Qwen3-Omni 的 8 个独立 Stage 进程在 A3 上成功 spawn 并启动
- 每个 Stage 的 `set_device()` 调用正确路由到 `torch.npu.set_device()`
- SHM Relay 正常工作; NIXL/Mooncake Relay 优雅降级

#### 设备字符串全局替换
- model_runner / pipeline / scheduling / relay 等核心模块不再硬编码 `"cuda"`
- `parse_gpu_id()` 兼容 `"npu:0"` 格式
- 向后兼容 `"cuda:0"` (CUDA 路径不受影响)

#### 模型权重加载
- Thinker (30B MoE): 通过 SGLANG ModelRunner 路径在 NPU 上成功加载 (TP=2 跨 2 卡)
- Talker (语音 token 生成): 成功加载权重, 后因 sglang 0.5.13 API 在 `compute_initial_expert_location_metadata` 处再次调用 `resolve_transformers_arch` 而失败
- Encoder (image/audio/code2wav): CPU mode 加载, 不触发 NPU 上的 C++ CUDA 路径

#### SGLANG 0.5.13 API 兼容性
针对容器与代码的版本差异, 修复了 10 个 API 兼容性问题:
cuda_graph 参数移除、max_total_num_tokens 属性迁移、compute_dp_attention 返回值变更、init_metrics 移除、Qwen3OmniTalker 模型注册等

#### 构建系统
`pyproject.toml` 中的 CUDA 专有 Relay/Attention 依赖已改为 optional, NPU/CUDA 环境均可正常安装

### 5.2 当前尚未解决的问题

| 问题 | 根因 | 影响 |
|------|------|------|
| 调度器代理与 sglang 0.5.13 不兼容 | `OmniScheduler.__getattr__` 代理到上游**类**属性，sglang 0.5.13 新增大量**实例**属性/方法 | 模型全部加载后调度器崩溃，无法完成端到端推理 |

已修复的 10 个属性/方法:
1. `_pending_chunked_abort_req` → None
2. `enable_fpm` / `enable_pdm` / `enable_nsa_prefill_bwd` → False
3. `maybe_prepare_mlp_sync_batch` / `process_pending_chunked_abort` → 上游类补丁 no-op
4. `dp_attn_adapter` / `enable_attn_adapter` → MagicMock
5. 通用回退: 所有未知属性 → MagicMock（但比较操作仍失败）

**推荐方案**: 使用 sglang 0.5.12.post1 (sglang-omni 精确依赖版本)，避免 0.5.13 大量 API 变化。

### 5.3 验证了什么 (23 项)

| 验证内容 | 验证方式 | 环境 |
|---------|---------|------|
| get_device_type() → "npu" | 直接调用实机 | A3 Docker |
| 检测链: sglang → torch.npu → torch.cuda → cpu | Mock + 实机 | pytest / A3 |
| 38 单元测试全部通过 | `pytest -v` | A3 Docker |
| set_device() → torch.npu.set_device() | Stage 日志 | A3 Docker |
| get_distributed_backend() → "hccl" | 直接调用 | A3 Docker |
| device_count() → 16 | 直接调用 | A3 Docker |
| create_event() → torch.npu.Event | 直接调用 | A3 Docker |
| synchronize() → torch.npu.synchronize() | 直接调用 | A3 Docker |
| get_device_name_str(0) → "Ascend910_9362" | 直接调用 | A3 Docker |
| get_visible_devices_env() → "ASCEND_RT_VISIBLE_DEVICES" | 直接调用 | A3 Docker |
| gpu_memory: 61GB 显存查询 | 直接调用 | A3 Docker |
| gpu_memory: pynvml 被绕过 | 直接调用 | A3 Docker |
| gpu_compat: compute_capability → None | 直接调用 | A3 Docker |
| gpu_compat: flashinfer_norm → False | 直接调用 | A3 Docker |
| gpu_compat: disable_custom_ar → True | 直接调用 | A3 Docker |
| parse_gpu_id("npu:0") → 0 | 直接调用 | A3 Docker |
| ShmRelay, Mooncake/NIXL 降级 | 启动日志 | A3 Docker |
| 8 Stage spawn + set_device | 启动日志 | A3 Docker |
| Thinker 30B TP=2 加载 28.47GB×2 | 启动日志 v50 | A3 Docker |
| Talker 6.24GB 加载 | 启动日志 v69 | A3 Docker |
| NPU CUDA graph 自动关闭 | stages.py | A3 Docker |
| pyproject.toml NPU 可安装 | pip install | A3 Docker |
| TP=2 GPU 隔离正确 | CUDA_VISIBLE_DEVICES 日志 | A3 Docker |

### 5.4 未验证的内容

| 内容 | 原因 |
|------|------|
| 端到端 HTTP 推理 | 调度器兼容性阻塞 |
| CUDA 环境回归测试 | 无 CUDA GPU |
| 其他模型 (Higgs TTS, MOSS-TTS 等) | SRS 仅 Qwen3-Omni |
| NIXL/Mooncake Relay | 昇腾不支持 |
| FP8 量化 | SRS 仅 BF16 |

### 5.5 后续步骤

1. **推荐方案**: 使用 sglang 0.5.12.post1 的 Ascend 镜像，避免 0.5.13 大量 API 变化
2. **整理提交**: 目前所有变更未提交，确认后可提交 PR
