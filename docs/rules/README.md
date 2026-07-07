# 代码库规范规则

> 由 codebase-scanner 于 2026-07-07 自动生成。
> 这些文档记录了项目现有的编码规范。
> 可自由编辑 — 下游技能在设计和开发阶段会读取这些文档。

## 文档

| 文档 | 说明 |
|------|------|
| [coding-style.md](coding-style.md) | 命名规范、格式化、文件组织、测试约定 |
| [coding-constraints.md](coding-constraints.md) | 二方/三方库约束、设备相关模式、静态分析工具、错误处理、导入规范 |
| [build-and-compilation.md](build-and-compilation.md) | 构建系统、CI/CD、打包、环境、pre-commit 钩子 |
| [commit-conventions.md](commit-conventions.md) | 提交格式、分支、PR、标签 |

## 关键发现摘要

- **语言**: Python（597 文件，173K 行），另有 C++/CUDA 内核代码
- **内部库（二方库）**: 7 个 — `vendor/sglang`、`sglang_backend`、`gpu_memory`、`gpu_compat`、`relay`、`profiler`、`client`
- **设备相关代码**: CUDA 硬编码遍布 — 21 个文件中 76 处 `torch.cuda` 调用点，30+ 文件中有 `"cuda:0"` 设备字符串，NVIDIA 专属依赖（nixl-cu13、mooncake-cuda13、pynvml）
- **现有硬件抽象**: Relay 基类（`sglang_omni.relay.base.Relay`）支持 SHM/NCCL/NIXL/Mooncake 后端；GPU 工具在 `gpu_memory`/`gpu_compat` 中（仅 CUDA）
- **静态分析工具**: Black（24.10.0）、isort（5.13.2）、Ruff（仅 F401）、autoflake（2.3.1）、clang-format（18.1.8）
- **构建系统**: setuptools + uv 包管理器，Python >=3.10
- **CI/CD**: GitHub Actions，自托管 H100/H200 GPU 运行器，标签触发运行
- **提交格式**: `[类别] 描述 (#PR编号)` — 80% 采用率；少量 Conventional Commits 用法

## 昇腾 A3 移植要点

以下方面在移植到昇腾 NPU 时需要重点关注：

1. **设备字符串模式**: 所有 `"cuda:X"` 字符串需要设备抽象层
2. **CUDA API 调用**: `torch.cuda.*`（76 处调用点）需要 NPU 等价实现（`torch.npu.*`）
3. **NCCL 依赖**: `relay/nccl.py` 硬编码 `dist.init_process_group("nccl", ...)` — 需要 HCCL 回退
4. **NVML 依赖**: `utils/gpu_memory.py` 有自定义 NVML 错误类和纯 NVML GPU 查询 — 需要 npu-smi 等价实现
5. **Docker 基础镜像**: `lmsysorg/sglang:dev` 是 x86_CUDA 专用 — 需要新的基础镜像
6. **Relay 依赖**: `nixl-cu13` 和 `mooncake-transfer-engine-cuda13` 仅支持 CUDA
7. **flash-attn-4**: 昇腾可能不可用；需要替代方案
8. **CUDA Graph 验证**: `utils/cuda_graph_batch_validator.py` 是 NVIDIA CUDA Graph 专有

---
*扫描器: codebase-scanner | 深度: deep | 采样文件: 200+（来自 597 Python 文件，173K 行）*
