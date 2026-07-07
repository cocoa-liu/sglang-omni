# 构建与编译

> 由 codebase-scanner 于 2026-07-07 自动生成。按需审核和调整。
> 来源: 仓库根目录配置文件。本文档无需源代码采样。
> **优先级**: pyproject.toml > Makefile/docs 配置 > CI workflow 配置。

## 构建系统

| 方面 | 详情 |
|------|------|
| 构建工具 | setuptools（>=61.0） |
| 构建后端 | `setuptools.build_meta` |
| 配置文件 | `pyproject.toml` |
| 包名称 | `sglang-omni` |
| 版本 | `0.1.0` |
| Python 要求 | >=3.10 |
| 许可证 | Apache-2.0 |

### 包发现

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["sglang_omni*"]
```

两个包被发布: `sglang_omni`（主包）和 `sglang_omni_router`（外部路由器，作为独立 CLI 安装）。

### 入口点

```toml
[project.scripts]
sgl-omni = "sglang_omni.cli:app"
sgl-omni-router = "sglang_omni_router.serve:main"
```

### 非 Python 数据文件

```toml
[tool.setuptools.package-data]
"sglang_omni.models.fishaudio_s2_pro" = ["fish_speech/configs/*.yaml"]
"sglang_omni.models.higgs_tts" = ["_vendored/*.json"]
```

## 关键构建/运行命令

| 命令 | 来源 | 说明 |
|------|------|------|
| `pip install -e .` 或 `uv pip install -e .` | pyproject.toml | 可编辑安装 |
| `pre-commit run --all-files` | `.pre-commit-config.yaml` | 运行所有 linter/格式化器 |
| `pytest tests/` | pyproject.toml `[tool.pytest.ini_options]` | 运行测试（pythonpath="."） |
| `sgl-omni serve ...` | 入口点 | 启动 Omni 服务器 |
| `sgl-omni-router ...` | 入口点 | 启动外部路由器 |

## 环境管理

| 方面 | 详情 |
|------|------|
| 包管理器 | **uv**（由 pyproject.toml `[tool.uv]` 段落和 `.gitignore` 中 `uv.lock` 条目确认） |
| uv 覆盖 | `protobuf>=6.31.1,<7.0.0` |
| 锁文件 | `uv.lock`（已 gitignore — 按环境重新生成） |
| Python 版本 | >=3.10（未找到 `.python-version` 或 `.tool-versions`） |

## 容器

| 方面 | 详情 |
|------|------|
| 基础镜像 | `lmsysorg/sglang:dev` |
| Dockerfile | `docker/Dockerfile` |
| Dockerignore | `.dockerignore` |
| 关键附加 | UCX 构建带 `--with-cuda=/usr/local/cuda` 用于 GPU-direct RDMA |
| TTS 基准 | `benchmarks/tts_serving/Dockerfile`（独立镜像） |

## 文档

| 方面 | 详情 |
|------|------|
| 构建 | `docs/Makefile`（Sphinx） |
| 依赖 | `docs/requirements.txt` |

## CI/CD

| 方面 | 详情 |
|------|------|
| 平台 | GitHub Actions |
| 工作流目录 | `.github/workflows/` |

### 工作流文件

| 工作流 | 用途 | 触发条件 |
|--------|------|---------|
| `omni-ci.yaml` | 主 Omni CI（多阶段流水线） | PR 标签 `run-ci` |
| `test.yaml` | 单元测试 | Push/PR |
| `test-asr-ci.yaml` | ASR 专用 CI | 基于标签 |
| `test-tts-ci.yaml` | TTS 专用 CI | 基于标签 |
| `test-qwen3-omni-ci.yaml` | Qwen3-Omni CI | 基于标签 |
| `test-layout.yaml` | 流水线布局验证 | Push/PR |
| `lint.yaml` | Lint 检查 | Push/PR |
| `docs-check.yaml` | 文档构建检查 | Push/PR |
| `publish-docs.yaml` | 发布文档到 GitHub Pages | 手动 / 合并到 main |
| `cancel-pr-workflow-on-merge.yaml` | 合并时自动取消 CI | PR 合并 |
| `cleanup-pr-ci-home-on-close.yaml` | PR 关闭时清理 CI home | PR 关闭 |
| `slash-command-handler.yml` | 处理 `/rerun-ci` 等 | Issue 评论 |

### CI 运行器

CI 运行在**自托管 GPU 运行器**（H100、H200）上。非维护者不能直接触发 CI — 维护者必须添加 `run-ci` 标签。参见 PR 模板。

### CI 自定义 Action

位于 `.github/actions/`:
- `omni-setup/` — 环境设置
- `omni-post-stage/` — 阶段后检查
- `omni-cleanup-host/` — GPU 状态清理
- `omni-save-cache/` — 缓存制品

## Pre-commit 钩子

**配置**: `.pre-commit-config.yaml`

**默认阶段**: `pre-commit`、`pre-push`、`manual`

| 钩子 | 工具 | 运行对象 |
|------|------|---------|
| 移除未使用导入 | autoflake v2.3.1 | 所有 Python（除 `.claude/skills/`） |
| 空格/合并/YAML/TOML | pre-commit-hooks v5.0.0 | 所有文件 |
| 导入排序 | isort 5.13.2 | 所有 Python（除 `.claude/skills/`） |
| 未使用导入检查 | ruff v0.11.10（仅 F401） | benchmarks/ docs/ examples/ |
| 格式化 | black 24.10.0（black-jupyter） | 所有 Python/notebooks（除 `.claude/skills/`） |
| C++ 格式化 | clang-format v18.1.8 | .cpp, .cu 文件 |
| 笔记本输出清除 | nbstripout 0.8.1 | Notebooks（保留输出，清除元数据） |
| CI 权限排序 | 本地脚本 | `.github/CI_PERMISSIONS.json` |
| 分支保护 | check no-commit-to-branch | 阻止提交到受保护分支 |

## 代码生成

未检测到 protobuf、OpenAPI、GraphQL 代码生成目录。

**内置代码**（不要修改，升级时可能需要重新生成）:
- `sglang_omni/vendor/sglang/` — 内置 SGLang 补丁（layers、models、distributed）

**已有模型内置代码**:
- `sglang_omni/models/higgs_tts/_vendored/` — Higgs 音频分词器配置

## 生成目录（排除在规范之外）

根据 `.gitignore`:
- `dist/`、`build/`、`*.egg-info/` — 构建产物
- `venv/`、`.venv/`、`env/` — 虚拟环境
- `__pycache__/`、`*.py[cod]` — Python 字节码
- `.pytest_cache/`、`.coverage`、`htmlcov/` — 测试产物
- `.ruff_cache/` — Lint 缓存
- `.benchmark-data/`、`results/` — 基准测试输出
- `.humanize/` — 性能分析输出
- `.tune-runs/`、`.eval-runs/` — 调优和评估输出

---
*扫描器: codebase-scanner | 深度: deep | 采样文件: 所有配置文件*
