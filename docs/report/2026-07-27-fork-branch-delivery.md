# Qwen3-Omni Ascend A3 fork 分支整理与验证记录

## 1. 目标

本记录描述如何把本地 NPU 适配代码、目标环境临时补丁和交付文档整理到
`git@github.com:cocoa00/sglang-omni.git` 的独立分支，并验证该分支能否在
Ascend A3 目标环境运行。

交付分支：`feat/qwen3-omni-ascend-a3`

## 2. 本地工作树审计

### 2.1 操作

检查本地分支、远端、提交基线和未提交内容：

```bash
git status --short --branch
git remote -v
git log -1 --oneline --decorate
git diff --stat
```

### 2.2 结果

- 本地原分支为 `main`，HEAD 为 `397de91`。
- 本地 HEAD 比当时的 `origin/main` 多 1 个文档规范提交。
- 工作树包含 28 个已跟踪代码修改，以及设备抽象、单测、报告和若干无关的
  未跟踪流程文件。
- 已跟踪代码修改全部属于本次 NPU 适配范围。
- 未跟踪文件不能整体执行 `git add -A`，否则会把自动化工具、模板和个人工作
  文件混入交付分支。

### 2.3 处理

代码提交仅包含：

- 28 个已跟踪的 NPU 适配文件；
- `sglang_omni/utils/device.py`；
- `tests/unit_test/utils/test_device.py`。

报告提交仅包含 `docs/report/`。其余未跟踪文件保持原状。

## 3. 目标环境补丁回收

### 3.1 第一次尝试：直接读取目标目录 Git 状态

尝试在容器中执行：

```bash
git -C /home/l00951280/sglang-omni status --short --branch
```

结果报错：目标目录不是可用的 Git 工作树。目标环境中的源码是部署副本，不能
依赖 `git diff` 回收补丁。

### 3.2 第二次尝试：导出源码树

先尝试同时导出 `sglang_omni`、`tests` 和 `docs/report`，但容器中不存在
`docs/report`，`tar` 返回 `Cannot stat`。

修正后只导出实际存在的目录：

```bash
docker exec lc-l3-test tar -C /home/l00951280/sglang-omni \
  -cf - sglang_omni pyproject.toml examples tests
```

源码树成功保存到本机 `/tmp`，随后与本地工作树逐文件比较。

### 3.3 比较结果

- 28 个本地已跟踪修改中，25 个与目标环境逐字一致。
- 差异集中在 `talker.py`、`code2wav_scheduler.py`、
  `request_builders.py` 和 `talker_model_runner.py`。
- `request_builders.py` 的有效逻辑已经与本地一致，差异仅为空行。
- `talker_model_runner.py` 的差异全部是定位问题时使用的一次性张量落盘。
- `talker.py` 同时包含运行修复和 residual logits 调试落盘。
- `code2wav_scheduler.py` 包含非流式请求整段解码补丁。

### 3.4 正式回收内容

回收以下运行逻辑：

- NPU Talker 使用 PyTorch 原生 temperature、top-k、top-p 和 multinomial
  采样，规避当前 BiSheng 无法编译 SGLang seeded sampler 的问题；
- Residual Code Predictor 使用参考采样参数；
- 修正 residual predictor 隐藏状态反馈方式；
- 非流式 Code2Wav 在请求结束后对完整 codec 序列统一解码。

不回收以下诊断内容：

- `/tmp/qwen3_omni_talker_trace_*.pt`；
- `/tmp/qwen3_omni_talker_full_codes_*.pt`；
- `/tmp/qwen3_omni_talker_prefill_input_*.pt`；
- `/tmp/qwen3_omni_first_residual_logits.pt`；
- 目标目录中的所有 `.bak-*` 文件。

原因是这些代码会在请求路径中无条件读写磁盘，不属于正式运行能力。

## 4. fork 和分支处理

### 4.1 GitHub 首次连接问题

第一次执行 `git ls-remote` 时出现 `Host key verification failed`。通过以下命令
记录 GitHub ED25519 主机指纹并验证 SSH 身份：

```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com
```

GitHub 返回身份 `cocoa00`，之后可正常读取 fork。

### 4.2 分支基线

fork 当时只有 `main`。新建：

```bash
git remote add fork git@github.com:cocoa00/sglang-omni.git
git switch -c feat/qwen3-omni-ascend-a3
```

核对发现当前适配基线比 `fork/main` 落后 92 个上游提交。由于所有 NPU 实机验证
都基于当前基线，本次不在交付前强行 rebase。这样可以保存实际运行版本，避免
把 92 个提交带来的冲突和未验证行为混入结果。后续同步上游应单独建分支完成。

## 5. 提交范围

代码提交：

```text
922ad92 feat(npu): adapt Qwen3-Omni pipeline for Ascend A3
```

提交包含 31 个文件，新增 1472 行、删除 207 行。未包含任何调试 trace 或备份
文件。

## 6. 验证过程

### 6.1 静态检查

- `git diff --check`：通过；
- 30 个涉及的 Python 文件执行 `py_compile`：通过；
- 搜索诊断文件名和 trace 标记：正式源码中无残留。

### 6.2 本机单测尝试

执行：

```bash
python3 -m pytest -p no:cacheprovider tests/unit_test/utils/test_device.py -q
```

本机返回 `No module named pytest`。这不是测试失败，而是当前工作机没有安装
pytest。目标容器已确认安装 pytest 8.3.2，因此单测改到目标容器执行。

### 6.3 目标机资源检查

检查时发现 NPU 0 上存在不属于本项目的进程：

```text
python -m sglang.launch_server --model-path /home/weights/MiniMax-M3-w8a8 ...
```

Qwen3-Omni 验证需要 NPU 0--8，因此不能重启容器、终止该进程或并发启动服务。
在资源释放前，只执行不占用 NPU 的导入、语法和单元测试。完整服务启动和固定
用例结果将在资源空闲后补充。

### 6.4 精确提交导入与单测

从代码提交生成 Git archive，传到目标机并解压到独立目录。两端 archive 的
SHA-256 一致，确认测试源码没有在传输中变化。

```text
commit: 922ad92
archive sha256: 128e78175097359ba84cfc4087ca611ca820352858bb124fc401c58ab88c66e8
```

通过 `PYTHONPATH` 指向独立目录后：

- `sglang_omni` 和 `sglang_omni.utils.device` 均从独立目录加载；
- 设备检测结果为 `npu`；
- 设备抽象单测结果为 `38 passed`，耗时 6.26 秒；
- Talker、Code2Wav 和 Talker 请求构建器导入通过。

第一次导入 Talker 时出现 `ModuleNotFoundError: No module named 'sgl_kernel'`。
检查发现目标容器安装的是 `sgl_kernel_npu 2026.6.1`，没有安装 CUDA
`sgl_kernel`；而 `thinker_model.py` 在模块顶层无条件导入
`fused_qk_norm_rope`，因此虽然代码已有非融合计算分支，仍会在进入该分支前
失败。

修复方式：

- 把 `sgl_kernel.fused_qk_norm_rope` 改为可选导入；
- 算子不存在时将 `compatible_with_fused_qk_norm_rope` 设为 false；
- 使用项目已有的 QK Norm 与 RoPE 非融合实现，张量仍在 NPU 上计算。

修复后 Talker、Code2Wav 和 Talker 请求构建器均能从精确提交源码成功导入。
另一次命令曾错误导入不存在的 `build_qwen3_omni_requests`，改为文件中真实存在
的 `build_sglang_talker_request` 后通过；该问题属于验证命令错误，不是代码
缺陷。

## 7. 已知边界

- 当前分支能够保存并复现本阶段 NPU 适配状态，但不是最新 `fork/main` 的适配。
- TTS 接口可返回格式合法的 WAV，人工试听仍为无意义语音，语音质量未通过。
- Image Encoder、Audio Encoder 和默认 Code2Wav 仍存在 NPU 环境下 CPU 降级。
- `mem_fraction_static=0.75` 是当前环境可启动参数，显存占用仍不符合 30B
  稀疏模型的合理预期。
- 显式 seed 的 NPU Talker 采样暂不支持，默认请求通过原生无 seed 采样规避
  BiSheng 编译错误。
