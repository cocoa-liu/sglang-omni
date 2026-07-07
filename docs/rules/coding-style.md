# 编码风格

> 由 codebase-scanner 于 2026-07-07 自动生成。按需审核和调整。
> 来源: 从 597 个 Python 文件（173K 行）中采样 200+ 文件
> **优先级**: 格式化器配置（black/isort）> 源代码观察。
> 与设计文档的冲突处会标注 "⚠ 覆盖" 注释。

## 命名规范

| 类别 | 模式 | 一致性 | 示例 |
|------|------|--------|------|
| 变量 | snake_case | 100% | `visible_devices`、`logical_gpu_id`、`pipeline_config` |
| 函数 | snake_case | 100% | `parse_cuda_visible_devices`、`build_sglang_server_args`、`validate_stage` |
| 方法 | snake_case | 100% | `setup_model`、`make_adapters`、`format_bytes_gib` |
| 类 | PascalCase | 100% | `TtsEngineBuilder`、`CudaGraphBatchReport`、`Stage`、`DecodeManager` |
| 模块 | snake_case | 100% | `gpu_compat.py`、`engine_factory.py`、`stage_workers.py` |
| 包 | snake_case | 100% | `sglang_omni`、`sglang_backend`、`audio_encoder` |
| 常量 | SCREAMING_SNAKE | 95% | `CUDA_VISIBLE_DEVICES`、`FLASHINFER_USE_CUDA_NORM`、`DEFAULT_TTS_BATCH_MAX_ITEMS` |
| 私有成员（模块级） | `_` 前缀 | 100% | `_EXPORTS`、`_shutdown_nvml`、`_try_import_pynvml`、`_get_device_handle` |
| 私有方法 | `_` 前缀 | 100% | `_find_available_port`、`_default_run_id`、`_stage_runtime_log_summary` |
| 布尔变量 | is/has/should 前缀 | 90% | `is_valid`、`is_available`、`has_attribute_override`、`should_disable_custom_all_reduce` |

## 格式化

| 规则 | 值 | 来源 |
|------|-----|------|
| 缩进 | 4 空格 | `.editorconfig:9` |
| 缩进风格 | 空格（禁用 Tab） | `.editorconfig:8` |
| 行尾 | LF（Unix） | `.editorconfig:7` |
| 编码 | UTF-8 | `.editorconfig:6` |
| 行尾空格 | 去除 | `.editorconfig:10` |
| 文件末尾换行 | 必须有 | `.editorconfig:11` |
| 行宽 | 88（black 默认） | `pyproject.toml`，pre-commit 中的 `[tool.black]` |
| 括号风格 | 同行（K&R/black） | 100% 采样文件中观察到 |
| 尾随逗号 | 多行时添加 | Black 自动格式化 |
| 引号风格 | 偏好双引号 | 约 80% 文件；含双引号的字符串用单引号 |
| 分号 | 从不使用 | 所有采样文件中 0 处 |
| 函数间空行 | 顶层 2 空行 | PEP 8，black 强制 |

**JSON/YAML**: 2 空格缩进（`.editorconfig:13-14`）。

## 格式化器配置

| 工具 | 配置文件 | 运行命令 |
|------|----------|---------|
| Black | `pyproject.toml`（black 24.10.0） | 通过 `pre-commit run black-jupyter` |
| isort | `.isort.cfg`（profile=black，known_first_party=sglang-omni） | 通过 `pre-commit run isort` |
| clang-format | `.editorconfig` + 内置样式 | `clang-format --style=file`（仅 C++/CUDA） |
| Ruff | 通过 `pre-commit`（仅 benchmarks/docs/examples 的 F401） | `ruff --select=F401 --fixable=F401` |

## 文件头规范

每个 `.py` 文件必须以以下内容开头：
```python
# SPDX-License-Identifier: Apache-2.0
```

**一致性**: 263 个采样文件中 263 个（约 100%）。部分文件紧接着添加模块级文档字符串。

## 模块文档字符串

约 80% 的 `__init__.py` 文件和约 60% 的模块文件包含描述模块用途的文档字符串。示例模式：
- `"""Stage — 流水线处理 IO 外壳。"""`
- `"""进程级 GPU 显存统计工具。"""`
- `"""配置驱动装配的导入工具。"""`

## `from __future__ import annotations`

**一致性**: 254/263 采样文件（约 97%）。启用 PEP 604 联合类型语法（`X | None` 代替 `Optional[X]`）和前向引用。所有新代码使用 `| None` 语法。

## 类型注解

| 方面 | 观察 |
|------|------|
| 使用频率 | 约 60% 函数/方法签名有类型注解 |
| 风格 | PEP 604 联合类型语法（`X | None`、`list[int]`） |
| TypedDict/dataclass | 大量使用（配置 `@dataclass(frozen=True)`） |
| `TYPE_CHECKING` | 约 5% 文件用于避免循环导入 |
| `Any` | 对第三方 API 对象自由使用（如 `server_args: Any`） |
| 严格检查 | 未找到 mypy.ini 或 pyrightconfig.json — 无严格类型检查 |

## 文件与目录组织

```
sglang_omni/                  # 主包
├── cli/                      # CLI 入口（typer）
├── client/                   # Omni API HTTP 客户端
├── config/                   # 流水线配置和放置逻辑
├── http/                     # HTTP 中间件（favicon、admin 认证）
├── model_runner/             # 模型执行器基类和补丁
├── models/                   # 各模型实现（按模型分目录）
│   ├── fishaudio_s2_pro/
│   ├── higgs_tts/
│   ├── llada2_uni/
│   ├── ming_omni/
│   ├── moss_transcribe_diarize/
│   ├── moss_tts/
│   ├── moss_tts_local/
│   ├── qwen3_asr/
│   ├── qwen3_omni/
│   ├── qwen3_tts/
│   ├── voxtral_tts/
│   └── whisper_asr/
├── pipeline/                 # 多进程流水线编排
│   └── stage/                # 阶段运行时、输入处理、流式
├── preprocessing/            # 音频/视频/图片预处理器
├── profiler/                 # 事件记录和性能分析
├── proto/                    # 类 protobuf 消息类型
├── relay/                    # 阶段间数据传输（NCCL、SHM、NIXL、Mooncake）
├── sampling/                 # 随机种子管理
├── scheduling/               # 请求调度器和 SGLang 后端集成
│   └── sglang_backend/       # SGLang 专用调度适配器
├── serve/                    # FastAPI 服务器、OpenAI API 适配器、WebSocket
│   ├── realtime/             # 实时 API（VAD、音频缓冲）
│   └── transcription_adapters/
├── utils/                    # 共享工具（GPU、音频、HF、导入）
└── vendor/                   # 内置 SGLang 补丁
    └── sglang/

sglang_omni_router/           # 外部路由器进程（独立包）
tests/                        # 测试（按模块共同放置）
├── unit_test/                # 单元测试（镜像 sglang_omni 结构）
│   ├── pipeline/
│   ├── scheduling/
│   ├── serve/
│   ├── model_runner/
│   └── ...
└── utils.py                  # 共享测试 fixtures 和 CI 工具
```

**组织模式**: 混合型 — 顶层按功能（models、scheduling、pipeline），模块内按层级（components/、pipeline/、stages）。

## 测试约定

| 方面 | 约定 |
|------|------|
| 位置 | 共同放置在 `tests/unit_test/<模块>/`，镜像源码结构 |
| 文件命名 | `test_<功能>.py` |
| 类命名 | 非必需；许多测试用纯函数配合 `@pytest.mark` |
| 初始化文件 | 每个测试包都有 `__init__.py` |
| 标记 | `@pytest.mark.gpu`（需 CUDA）、`@pytest.mark.benchmark`、`@pytest.mark.tts_stage(name)` |
| 异步测试 | `pytest-asyncio` 配合 `async def` 函数 |

---
*扫描器: codebase-scanner | 深度: deep | 采样文件: 200+*
