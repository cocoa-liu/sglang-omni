# 提交规范

> 由 codebase-scanner 于 2026-07-07 自动生成。按需审核和调整。
> 来源: `git log --oneline -100`、`git branch -r`、`.github/`

## 提交消息格式

**主导模式**: 主题标签式规范提交（非严格 Conventional Commits 规范）。

### 格式

```
[类别] 简短描述 (#PR编号)

可选的正文，含更多上下文。
```

### 观察到的类别（最近 100 次提交）

| 标签 | 数量 | 示例 |
|------|------|------|
| `[CI]` | 24 | `[CI] Clean up GPU state after ASR stage startup failure (#958)` |
| `[Perf]` | 8 | `[Perf] Defer Omni stage factory imports to worker startup (#900)` |
| `[TTS]` / `[TTS Refactor]` | 6 | `[TTS] Validate generation batch policies (#836 W1) (#843)` |
| `[Qwen3-Omni]` | 4 | `[Qwen3-Omni] Lazy TensorRef path for large inter-stage tensors (#797)` |
| `[MOSS]` / `[Moss]` | 6 | `[Moss] Increase MOSS-TTS Local ref audio cache item cap (#788)` |
| `[ASR]` | 3 | `[ASR] Default transcription requests to greedy decoding (#969)` |
| `[RL]` | 2 | `[RL] distributed weight-sync (#784)` |
| `[Higgs]` | 2 | `[Higgs]: Refine Higgs decode mode CLI (#700)` |
| `[Feat]` | 2 | `[Feat] Moss tts local streaming (#753)` |
| `[Docs]` | 2 | `[Docs] Internalize RFC comments + consolidate historical RFCs` |
| `[router]` | 1 | `[router] Stream proxied responses instead of buffering` |
| `[Misc]` | 1 | `[Misc]: Use direct access for strict fields` |
| `[Logging]` | 1 | `[Logging]: Use f-strings for simple logging calls` |
| `docs:` | 2 | `docs: update whisper_asr default temperature to 0.0 (#974)` |
| `perf(...):` | 2 | `perf(moss-td): batch whisper encoder across requests (#971)` |
| `feat(...):` | 2 | `feat(higgs/rl): add rollout logprob + delay-pattern action-mask kernels (#823)` |
| `fix(...):` | 2 | `fix(realtime): define audio buffer overflow error (#706)` |

### 格式分析

| 指标 | 值 |
|------|-----|
| 主导格式 | `[标签] 消息 (#PR)` — 80% |
| Conventional Commits（`type(scope):`） | 10% |
| 标题行长度（P95） | 约 70 字符 |
| 正文使用 | 约 40% 提交有正文 |
| 页脚模式 | 罕见；主要是 Co-authored-by 或 Fixes #N |
| PR 引用 | 总是追加为 `(#NNN)` |

## 分支命名

| 模式 | 示例 | 频率 |
|------|------|------|
| `main` | `origin/main` | 主分支 |
| `claude/fervent-cannon-XXXXX` | `origin/claude/fervent-cannon-4hQa6` | 25+ 分支（Claude Code worktree） |
| `feature-name` | `origin/ASR_eval`、`origin/calibration`、`origin/clean_ci_up` | 约 10 分支 |
| `feature/description` | `origin/add-audio-understanding-benchmarks` | 约 5 分支 |

无严格分支命名规范；`claude/*` 前缀分支是自动生成的 worktree 分支。

## PR 规范

**PR 模板**: `.github/pull_request_template.md` 存在。

模板段落:
1. **Motivation** — 目的和目标
2. **Modifications** — 所做的更改
3. **Related Issues** — 关联 Issue（如 "Fixes #123"）
4. **Accuracy Test** — 模型侧代码
5. **Benchmark & Profiling** — 性能影响
6. **Checklist** — 格式化代码、单元测试、文档、基准测试
7. **CI** — 自托管 GPU 运行器，需要维护者添加 `run-ci` 标签

## 变更日志

未找到 `CHANGELOG.md`。发布版本通过 git 标签和 GitHub Release Notes 跟踪。

## 标签与发布

| 方面 | 详情 |
|------|------|
| 标签格式 | `0.1.0`（语义化版本，无 `v` 前缀） |
| 最新标签 | `0.1.0` |
| 标签数量 | 1（项目处于 pre-1.0 阶段） |

## 其他

| 方面 | 详情 |
|------|------|
| Signed-off-by | 罕见（非必需） |
| Co-authored-by | 偶尔出现 |
| `Fixes #N` | 部分 PR 使用 |
| Breaking change 标记 | 当前历史中未观察到 |

---
*扫描器: codebase-scanner | 深度: deep | 来源: git log -100, git branch -r, git tag*
