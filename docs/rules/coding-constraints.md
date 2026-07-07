# 编码约束

> 由 codebase-scanner 于 2026-07-07 自动生成。按需审核和调整。
> 来源: 从 597 个 Python 文件（173K 行）中采样 200+ 文件
> **优先级**: 框架要求 > Linter 配置 > 源代码观察。

## 二方（内部）库检测

| 领域 | 内部库 | 替代 | 导入模式 | 频率 |
|------|--------|------|---------|------|
| SGLang 后端 | `sglang_omni.vendor.sglang` | 直接 `sglang` 导入 | `from sglang_omni.vendor.sglang import ...` | 7 文件 |
| SGLang 调度 | `sglang_omni.scheduling.sglang_backend` | 直接 SGLang 调度器调用 | `from sglang_omni.scheduling.sglang_backend import ...` | 8 文件 |
| GPU 显存 | `sglang_omni.utils.gpu_memory` | 直接 `pynvml`/`torch.cuda` 调用 | `from sglang_omni.utils.gpu_memory import ...` | 12 文件 |
| GPU 兼容性 | `sglang_omni.utils.gpu_compat` | 直接 SM 版本检查 | `from sglang_omni.utils.gpu_compat import ...` | 5 文件 |
| Relay/传输 | `sglang_omni.relay` | 直接 `torch.distributed` send/recv | `from sglang_omni.relay import ...` | 15 文件 |
| 性能分析 | `sglang_omni.profiler` | 直接 `torch.profiler` | `from sglang_omni.profiler import ...` | 10 文件 |
| 客户端 | `sglang_omni.client` | 直接 `httpx`/`requests` | `from sglang_omni.client import ...` | 20 文件 |

**惰性加载 __init__.py 模式**: 多个包使用 `_EXPORTS` 字典 + `__getattr__` 模式实现延迟导入：
```python
_EXPORTS = {"ExportName": ("sglang_omni.模块.子模块", "ClassName")}
def __getattr__(name): ...  # 首次访问时延迟导入
```
使用位置: `sglang_omni/config/__init__.py`、`sglang_omni/pipeline/__init__.py`、`sglang_omni/utils/__init__.py`、`sglang_omni/pipeline/stage/__init__.py`。

## 三方库约束

| 领域 | 所选库 | 版本 | 锁定方式 |
|------|--------|------|---------|
| HTTP 服务器 | FastAPI + Uvicorn | >=0.110.0 | 范围 |
| 数据验证 | Pydantic | >=2.0.0 | 范围 |
| 序列化 | msgpack | >=1.0.0 | 范围 |
| 日志 | 标准库 `logging` | stdlib | N/A |
| 测试 | pytest | >=7.0.0 | 范围 |
| 异步测试 | pytest-asyncio | >=0.21.0 | 范围 |
| YAML 配置 | PyYAML | >=6.0 | 范围 |
| CLI | Typer | >=0.9.0 | 范围 |
| 音频处理 | librosa + soundfile | >=0.11.0 / >=0.12.0 | 范围 |
| 视频解码 | torchcodec | ==0.11.1 | 精确锁定 |
| ML 框架 | torch | ==2.11.0 | **精确锁定** |
| torchvision | torchvision | ==0.26.0 | **精确锁定** |
| torchaudio | torchaudio | ==2.11.0 | **精确锁定** |
| SGLang | sglang | ==0.5.12.post1 | **精确锁定** |
| Transformers | transformers | ==5.6.0 | **精确锁定** |
| 注意力机制 | flash-attn-4 | >=4.0.0b9,<4.0.0b16 | 范围（测试版） |
| 内核 | kernels | >=0.14.0,<0.15 | 范围 |
| 阶段间传输（CUDA） | nixl-cu13 | >=1.1.0 | 范围（CUDA 13 专用） |
| 阶段间传输（CUDA） | mooncake-transfer-engine-cuda13 | >=0.3.10 | 范围（CUDA 13 专用） |
| 包管理器 | uv | 隐式（pyproject.toml `[tool.uv]`） | N/A |

**关键观察**:
- torch/sglang/transformers 精确锁定版本 — 依赖升级必须协调进行
- nixl 和 mooncake 使用 CUDA 13 专用 wheel（cu13 后缀） — 非 CUDA 后端需要替代 relay 实现
- Logger 库依赖（`"logger"`）在 PyPI 上，但用途不明确 — 标准库 `logging` 模块被普遍使用

## 设备相关代码模式（CUDA 硬编码）

⚠ **昇腾移植关键**: 代码库中存在大量 CUDA/NVIDIA 专有硬编码。

| 模式 | 影响文件 | 示例 |
|------|---------|------|
| `device = "cuda:0"` 默认值 | 30+ 文件 | `engine_factory.py:26`: `device: str = "cuda:0"` |
| `torch.cuda.is_available()` | 21 文件，76 处 | 遍布各处 GPU 操作守卫 |
| `torch.cuda.device_count()` | 用于 `gpu_memory.py`、`gpu_compat.py` | 设备数量查询 |
| `torch.cuda.current_device()` | `relay/nccl.py`、`stage_workers.py` | 当前设备设置 |
| `torch.cuda.set_device()` | `relay/nccl.py:50`、`misc.py:43` | 设备绑定 |
| `f"cuda:{gpu_id}"` 字符串 | 30+ 文件 | 设备字符串构造 |
| `dist.init_process_group("nccl", ...)` | `relay/nccl.py:54-58` | NCCL 后端硬编码 |
| `torch.cuda.get_device_properties()` | `gpu_memory.py:200`、`gpu_compat.py:60` | SM 版本查询 |
| `torch.cuda.mem_get_info()` | `misc.py:51` | 显存查询 |
| pynvml（NVIDIA NVML） | `gpu_memory.py`、`gpu_compat.py` | GPU 管理库（无昇腾等价物） |
| CUDA_VISIBLE_DEVICES 环境变量 | `gpu_memory.py:34`、`gpu_compat.py` | 设备可见性解析 |
| nixl-cu13 / mooncake-cuda13 | `pyproject.toml:33-34` | 仅 CUDA relay 依赖 |
| Docker CUDA 路径 | `docker/Dockerfile:16` | `--with-cuda=/usr/local/cuda` |

**现有抽象模式**:
- `sglang_omni.utils.gpu_memory` / `gpu_compat`: 集中的 GPU 工具（但仅 CUDA）
- `sglang_omni.relay`: 抽象 `Relay` 基类，配合 NCCL/NIXL/SHM/Mooncake 后端 — 此模式可扩展
- `TtsEngineBuilder.build()` 中的 `device: str` 参数可接受任意设备字符串（如 "cuda:0"），但下游代码经常硬编码 "cuda"

## 禁止使用的 API / 库（推断）

| 禁止/缺失 | 证据 |
|-----------|------|
| `print()` 日志输出 | 生产代码中从未使用；始终用 `logging.getLogger(__name__)` |
| 裸 `open()` 文件操作 | 始终用 `pathlib.Path`，未发现直接 `open()` |
| `requests` 库 | 不在依赖中；HTTP 使用 `httpx` |
| `pickle` 跨进程通信 | msgpack + ZMQ 用于消息传递；`pickle` 仅在 `broadcast_pyobj` 工具中 |

## 静态分析工具

| 工具 | 配置文件 | 运行命令 | 范围 |
|------|---------|---------|------|
| Black（格式化） | pyproject.toml | `pre-commit run black-jupyter` | 所有 Python |
| isort（导入排序） | `.isort.cfg` | `pre-commit run isort` | 所有 Python |
| Ruff（linter，仅 F401） | 通过 pre-commit | `pre-commit run ruff` | 仅 benchmarks/ docs/ examples/ |
| autoflake（未使用导入） | 通过 pre-commit | `pre-commit run autoflake` | 所有 Python（除 `.claude/skills/`） |
| clang-format | 通过 pre-commit | `clang-format --style=file` | 仅 C++/CUDA |
| nbstripout | 通过 pre-commit | `pre-commit run nbstripout` | Jupyter 笔记本 |
| pre-commit-hooks | 通过 pre-commit | `pre-commit run` | 空格、YAML、TOML、合并冲突 |

**注意**: 未找到 mypy、pyright、flake8 配置文件。项目目前不通过静态分析强制执行严格类型检查。

## 错误处理模式

| 方面 | 模式 |
|------|------|
| 主导模式 | try/except 配合特定异常类型 |
| 自定义异常 | `gpu_memory.py` 中的 `_InvalidGpuDeviceError(RuntimeError)` |
| 异常链 | 持续使用 `from exc` |
| 错误日志级别 | 预期失败用 `logger.debug`，降级用 `logger.warning`，严重用 `logger.error` |
| 集中处理 | 无全局异常处理器；各组件自行处理 |
| 失败返回 None | 常见模式：GPU/NVML 不可用时函数返回 `None` |

## 导入组织

**观察到的顺序**（95%+ 文件一致）：

1. Future 导入: `from __future__ import annotations`
2. 标准库（按字母排序）
3. 三方库导入（按字母排序）
4. `sglang_omni` 内部导入（绝对导入，从顶层）
5. 每组之间空一行

**所有导入为绝对导入**，使用 `from sglang_omni.X import Y`。在 `sglang_omni/` 包中零相对导入（`from . import` / `from .. import`）。

**例外**: 测试文件有时混用绝对导入（`from sglang_omni.X import Y`）和测试包内相对导入。

## 注释/文档风格

| 方面 | 观察 |
|------|------|
| 文档字符串格式 | reStructuredText / Google 风格混合（Args/Returns 段落） |
| 公共函数覆盖率 | 约 60% |
| 位置 | 紧接 `def` 行下方 |
| 段落标题 | `"""单行概述。\n\n详细描述。\n\nArgs:\n    ...\nReturns:\n    ..."""` |
| 行内注释 | `# ` 带空格，用于笔记和 TODO |

## 测试约定

| 方面 | 约定 |
|------|------|
| 框架 | pytest（>=7.0.0） |
| 异步支持 | pytest-asyncio（>=0.21.0） |
| 断言风格 | `assert` 语句（非 `assertEqual`） |
| 测试分组 | 扁平函数配合 `test_` 前缀；最小化类使用 |
| Fixtures | `pytest.fixture` 在模块级别 |
| Mock 框架 | `unittest.mock`（标准库），通过 pytest  monkeypatch |
| CI 标记 | `@pytest.mark.gpu`、`@pytest.mark.benchmark` |

---
*扫描器: codebase-scanner | 深度: deep | 采样文件: 200+*
