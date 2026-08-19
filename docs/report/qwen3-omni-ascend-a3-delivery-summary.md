# Qwen3-Omni Ascend A3 NPU 适配与问题修复报告

日期：2026-08-19

## 1. 结论

- `feat/qwen3-omni-ascend-a3` 已完成 Qwen3-Omni 从 CUDA 专属路径到 Ascend NPU 可选运行路径的适配。
- 分支基线 `d6c98e55` 已包含 TTS 乱码的两项核心修复；本次在该基线上补充了辅助组件设备契约和 Talker 推理 no-grad 边界。
- 原先观察到的高 Ready HBM 主要来自 SGLang 按剩余显存建立的 auto-KV 池，不能单独判定为显存泄漏。本次没有加入 KV cap、固定显存阈值或物理卡号硬编码。
- 历史实机功能与精度结果可作为实现方案的支持证据；按照本次交付决定，最终分支新增提交未重复执行完整实机测试，因此不能把历史结果表述为最终 HEAD 的直接复测结果。

## 2. 版本与提交范围

| 项目 | 提交 | 说明 |
| --- | --- | --- |
| 仓库主分支基线 | `3f97d73bc10ad4048b5a202cf586457350cee7d5` | 本 PR 的目标基线 |
| NPU 适配主提交 | `922ad92` | 增加 Ascend 设备抽象、权重加载、通信与 Qwen3-Omni NPU 运行路径 |
| TTS 语义修复 | `322fc18` | 修正 residual codec embedding 累加和 greedy 采样语义 |
| TTS MoE 修复 | `d6c98e55` | NPU Talker 使用与权重布局一致的 Triton MoE backend |
| 辅助组件设备契约 | `d186600` | placement 决定设备；禁止静默回退 CPU；增加设备断言和测试 |
| Talker 显存修复 | `982308b` | `code_predictor_forward` 增加 `@torch.no_grad()` 及回归测试 |

本次新增代码相对 `d6c98e55` 为 8 个文件、`+425/-31`。其中约 129 行为测试，约 122 行为通用设备校验与诊断；核心行为变化是设备归属 fail-fast 和一行 no-grad 边界。

整个 feature 分支相对 `main` 还包含此前完成的 NPU 适配、TTS 修复及相关文档，因此 PR 总 diff 大于本次两个新增提交。

## 3. 三个问题的根因与处理

| 问题 | 根因 | 处理 | 当前状态 |
| --- | --- | --- | --- |
| TTS 乱码 | NPU 适配过程中，residual code predictor 将中间 transformer hidden state 错误加入下一帧反馈，并把参考 greedy 路径改成随机采样；同时 Talker MoE backend 的权重布局与 NPU 执行约定不一致 | `322fc18` 恢复 codec embedding 累加和 greedy 语义；`d6c98e55` 选择与 NPU 权重布局一致的 Triton backend | 已在本次基线中修复，不重复提交或改写作者历史 |
| 关键组件静默落 CPU | image encoder、audio encoder 和 Code2Wav 在缺失明确设备时存在 NPU→CPU 默认回退；image/audio stage factory 还会忽略 placement 的 `gpu_id` 并使用固定 logical device 0 | placement 作为唯一设备来源；缺失或冲突直接失败；启动时检查全部 parameter/buffer，首个真实 forward 检查输入和原始输出 | 本次新增修复 |
| 显存异常 | 高 Ready HBM 主要是 auto-KV 预留，不等同于泄漏；独立请求级增长来自 Talker code predictor 推理时仍开启 autograd，导致计算图保留 | 保留 auto-KV 语义；在共享 inference-only 入口 `code_predictor_forward` 增加 `@torch.no_grad()` | 本次新增修复；没有通过限制 KV 容量掩盖问题 |

### 3.1 TTS 乱码

`322fc18` 修复了两个由 NPU 适配引入的语义偏差：

1. 每个 residual group 都应把新生成 codec code 的 embedding 加入 `summed_embeddings`；transformer hidden state 只负责驱动下一组自回归计算，不能作为 codec feedback。
2. 当前参考配置是 `do_sample=False`，因此 layer-0 和 residual code 均使用 greedy argmax，而不是 NPU 专用 multinomial 分支。

`d6c98e55` 进一步修复 Talker MoE。原 backend 按 `[up, gate]` 加载权重，但 NPU kernel 按 `[gate, up]` 解释，导致第一层 expert 输出已经偏离参考实现。NPU Talker 改用 Triton backend 后，加载布局与执行 kernel 一致。

这两项修复已经属于目标分支历史，本次只在报告中说明，不重复提交代码。

### 3.2 辅助组件 CPU 降级

修改涉及：

- `components/common.py`：新增统一的 concrete device 解析、参数/buffer 检查和 tensor tree 检查。
- `stages.py`：image/audio factory 接受运行时根据 placement 注入的 `gpu_id`，不再自行猜测设备。
- `components/image_encoder.py`、`audio_encoder.py`、`code2wav_scheduler.py`：删除 NPU 环境下的默认 CPU 回退；加载后验证模型设备，首个真实 forward 验证输入与原始输出设备。
- `test_component_device_contract.py`：覆盖缺失设备、显式 CPU、placement 冲突、参数/buffer 错配、tensor 错配和 factory 签名。

设计原则是：

```text
placement logical slot
        -> runtime 注入 gpu_id
        -> factory 解析 concrete device
        -> component 校验并执行
```

组件不写宿主物理卡号，不猜测 `npu:0`，也不会在配置错误时回退 CPU 后继续返回成功。显式 CPU 模式仍可由调用者主动指定，删除的是静默降级，不是删除 CPU 工具链。

### 3.3 显存问题

显存结论必须区分两类现象：

1. **正常容量预留**：SGLang 根据 `mem_fraction_static` 和加载后剩余显存建立 KV pool。高 Ready HBM 若能由权重、KV、workspace 账本解释，并在进程退出后回收，不属于泄漏。
2. **请求级 autograd retention**：Talker 在推理路径调用 `code_predictor_forward` 时，外层并非始终处于 no-grad 范围。该函数会执行 residual predictor 并返回 tensor；如果 autograd 开启，会形成不必要的计算图并造成请求相关显存增长。

最终修复仅在 inference-only 共享入口增加：

```python
@torch.no_grad()
def code_predictor_forward(...):
    ...
```

没有采用以下已否决方案：

- 不把 `max_total_tokens` 固定为单上下文容量；
- 不降低 `mem_fraction_static` 来制造较低 HBM 数字；
- 不设置 24 GiB/rank 等经验阈值；
- 不要求 9 个 logical NPU；
- 不在源码中固定宿主物理卡号；
- 不通过每请求 `empty_cache()` 掩盖对象生命周期问题。

## 4. 已有功能与精度证据

历史同条件 A/B 使用相同权重、数据、评分器和 4 logical NPU / Thinker TP2 / auto-KV 配置；没有设置 `max_total_tokens`、`mem_fraction_static` 覆盖或 KV cap。

对照对象为官方版本（已包含 PR #1476）与此前的同源验证候选 `7b0f1b133b5001b524b97a37ebff32a26b713c4d`。这些结果证明修复方案在覆盖范围内没有超过预设门禁的精度退化，但不是最终 `d6c98e55` 派生 HEAD 的直接复测。

| 指标 | 官方结果 | 验证候选结果 | 结论 |
| --- | ---: | ---: | --- |
| SeedTTS 有效生成 | 1088/1088 | 1088/1088 | 双方完整完成 |
| SeedTTS corpus WER | 1.8756% | 2.2272% | 高 0.3517 个百分点，通过预设 `<= +0.5pp` 门禁 |
| MMSU accuracy | 70.75% | 70.80% | 通过 |
| MMMU-CI accuracy | 62% | 60% | 少 1/50 题，通过 54% 绝对回归门禁；仅作为小样本 canary |
| ASR 评分器参考音频 WER | 1.2182% | 同一评分器 | 通过预设 `<= 1.5%` 健康门禁 |

另有以下支持证据：

- 固定多模态功能集合完成 31 个预期成功请求和 1 个精确预期错误。
- 30 分钟压力测试完成 3440/3440 个请求。
- Talker no-grad 修复前后 framework A/B 的 20/20 门禁通过：修复后排空请求并同步时 `memory_allocated` 不再保留请求计算图增量。
- 受影响单元测试曾在同源验证候选上通过；skipped 项未计为 passed。

精度证据包 SHA-256：

```text
0ae49e07616d014ab6d6ce7071bc4ff5b93af5168026be3837b10e4c4ec6d728
```

## 5. 本次最终分支的验证状态

按照本次交付决定，不再为最终分支重复完整实机实验。提交前已完成的静态检查为：

| 项目 | 结果 | 状态 |
| --- | --- | --- |
| `git diff --check` | 无 whitespace error | 通过 |
| 修改文件 Python 语法编译 | 无语法错误 | 通过 |
| 新增聚焦单元测试 | 已随代码提交 | 未在最终分支重新执行 |
| 最终 HEAD NPU 短闭环 | 未执行 | 未验证 |
| 最终 HEAD 1088 条精度 A/B | 不重复执行 | N/A |

因此，可以主张：问题根因、最小修改及历史同源验证证据已经提交；不能主张最终 HEAD 已重新完成全部实机回归。

## 6. 风险与边界

1. 辅助组件修复会把原先可能静默成功的错误配置改为启动失败，这是预期的 fail-fast 行为，但最终分支未做实机复测，仍可能存在未覆盖的 factory 参数或 runtime 兼容问题。
2. no-grad 修改只有一行，且不改变 logits、采样或权重；风险较低，但最终分支未重复运行内存门禁。
3. 官方和验证候选在多样化并发 8 TTS 中都出现过共同调度故障，不能声称任意并发 TTS 均正常。
4. 曾尝试的公共 batchfilter 补丁虽然生成 1088/1088 个 WAV，但 WER 达到 94.38%，已经否决且未提交。
5. 精度结果只能说明覆盖指标通过预设门禁，不能表述为与官方完全相同、更优或统计等价，也不覆盖 MOS、全部语言和全部拓扑。

## 7. 审阅建议

建议按提交顺序审阅：

1. `d186600`：检查 placement 是否成为 image/audio/Code2Wav 的唯一默认设备来源，以及所有错误是否 fail-fast。
2. `982308b`：检查 `code_predictor_forward` 是否确属 inference-only 共享入口及测试是否验证计算图关闭。

两个修复提交相互独立；若后续发现兼容性小问题，可以在对应提交范围内修正，不需要回退已有的 TTS 乱码修复或重新引入 KV 资源限制。
