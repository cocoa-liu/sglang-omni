# Qwen3-Omni TTS NPU/NPUIR 问题调查记录

日期：2026-07-21

目标环境：
- 远程配置：`/home/cocoa/lc/scripts/remote.json`
- 远程主机：`113.46.38.25`
- Docker 容器：`lc-l3-test`
- 远端代码目录：`/home/l00951280/sglang-omni`
- 模型路径：`/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct`

目标：
- 找出 Qwen3-Omni TTS 在昇腾 NPU 上失败的真实根因。
- 不盲信已有报告，以可复现命令和实测日志为准。
- 当前 NPU 卡被占用时，先做不加载 Qwen3-Omni 的小张量复现。

## 步骤 1：远程连通性和环境检查

目的：
- 确认目标机器和 Docker 容器可访问。
- 在任何复现前确认运行时包版本。

命令概要：
- 根据 `/home/cocoa/lc/scripts/remote.json` SSH 到 `root@113.46.38.25`。
- 在 `lc-l3-test` 容器内运行 Python，打印 Python 路径、`torch`、`torch_npu`、`sglang`、NPU 数量和 `sglang_omni` 导入路径。

观察结果：
- 容器 `lc-l3-test` 正在运行。
- Python 路径：`/usr/local/python3.11.15/bin/python3`
- `torch`：`2.10.0+cpu`
- `torch_npu`：`2.10.0`
- NPU 数量：`16`
- `sglang`：`0.5.12.post1`
- `sglang_omni`：`/home/l00951280/sglang-omni/sglang_omni/__init__.py`

遇到的问题：
- 本地 SSH 首次失败，报错：
  `Bad owner or permissions on /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf`

解决方式：
- 使用 `ssh -F /dev/null` 绕过本机系统 SSH 配置后重试成功。

## 步骤 2：确认当前是否适合完整复现 Qwen3-Omni

目的：
- 避免干扰正在占用机器的任务。
- 判断当前是否可以启动完整 Qwen3-Omni 服务。

命令概要：
- 检查容器内进程。
- 通过 `npu-smi info` 检查 NPU 显存占用和进程归属。

观察结果：
- 另一个 SGLang DeepSeek 服务正在占用全部 16 张 NPU：
  `/data/weights/DeepSeek-V4-Flash-w8a8-mtp`，参数包括 `--tp-size 16`、`--dp-size 16`，端口 `9527`。
- 每张 NPU 大约占用 40 GB HBM。
- `lc-l3-test` 容器内存在旧的 `sgl-omni`/`python3.11` 僵尸进程，但没有发现活跃的 Qwen3-Omni 服务。

遇到的问题：
- 当前 NPU 显存不够空，不能安全启动完整 Qwen3-Omni 服务复现。

解决方式：
- 完整 Qwen3-Omni 复现推迟到用户确认卡释放后再做。
- 当前只继续做小张量 NPU 复现。

## 步骤 3：裸 PyTorch 小张量算子复现

目的：
- 检查可疑基础算子在 NPU 上单独运行是否失败。
- 避免在机器被占用时加载 Qwen3-Omni。

命令概要：
- 通过 SSH 管道把一个小 Python 脚本传入：
  `docker exec -i lc-l3-test python3 -`
- 创建：
  `logits = torch.randn(2, 1024, device="npu:0", dtype=torch.float32)`
- 测试以下操作：
  - `torch.argmax(logits, dim=-1)`
  - `torch.softmax(logits, dim=-1)`
  - `torch.topk(logits, k=50, dim=-1)`
  - `torch.argmax(...).to(torch.uint32)`
  - Gumbel-Max：`rand_like -> clamp -> log -> log -> add -> argmax`

观察结果：
- 所有裸 PyTorch 操作都在 `npu:0` 上成功运行。
- 仅出现一个 warning：
  `Cannot create tensor with interal format while allow_internel_format=False, tensor will be created with base format.`

遇到的问题：
- 本地沙箱拦截了通过管道执行的 SSH 命令，报错：
  `socket: Operation not permitted`

解决方式：
- 使用已批准的 SSH 提权权限重跑同一条命令。

当前判断：
- NPUIR 失败不太可能由这些基础算子单独触发。
- 触发点更可能是更大的组合图、SGLang sampler 控制路径，或 Talker 模型/预测器图。

## 步骤 4：第一次尝试 SGLang Sampler 小张量复现

目的：
- 用小张量运行真实的 `sglang.srt.layers.sampler.Sampler.forward`。
- 对比 `sampling_backend="pytorch"` 和 `sampling_backend="ascend"`。
- 对比 greedy、simple multinomial、top-k/top-p 三条路径。

命令概要：
- 使用 `Sampler.__new__(Sampler)` 并手工设置属性，避免初始化分布式 TP group。
- 尝试用下面方式初始化 SGLang 全局参数：
  `ServerArgs(model_path="/tmp/dummy", device="npu", sampling_backend=backend)`

观察结果：
- 所有 case 都在进入 sampler 前失败。
- 报错：
  `OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/tmp/dummy'.`

遇到的问题：
- `ServerArgs.__post_init__()` 会处理模型配置，并尝试为 `/tmp/dummy` 加载 HuggingFace config。
- 因此 `ServerArgs` 不适合做“不加载模型”的 sampler-only 复现。

解决方式：
- 后续不再用 `ServerArgs` 做这个小测试。
- 改为直接向 `sglang.srt.server_args._global_server_args` 注入一个最小对象，只包含 sampler 代码需要的字段。

## 步骤 5：使用最小全局参数复现 SGLang Sampler

目的：
- 不触发模型 config 加载，重新运行 `Sampler.forward`。
- 验证 SGLang sampler 自身是否能在小 logits 上复现 NPU/NPUIR 失败。

命令概要：
- 注入：
  `sglang.srt.server_args._global_server_args = SimpleNamespace(...)`
- 用 `Sampler.__new__(Sampler)` 创建 `Sampler`，并手工设置 `forward` 会用到的属性。
- 创建：
  `logits = torch.randn(2, 1024, device="npu:0", dtype=torch.float32)`
- 测试：
  - backend `pytorch`，mode `greedy`
  - backend `pytorch`，mode `simple`
  - backend `pytorch`，mode `topk_topp`
  - backend `ascend`，mode `greedy`
  - backend `ascend`，mode `simple`
  - backend `ascend`，mode `topk_topp`

观察结果：
- 6 个 case 全部在 `npu:0` 上成功。
- 返回 tensor 保持在 NPU 上。

遇到的问题：
- 本轮没有遇到问题。

解决方式：
- 无需处理。

当前判断：
- SGLang sampler 不会仅仅因为“小张量 top-k/top-p 采样”而失败。
- 完整 TTS 失败可能需要更接近真实的 shape、dtype、deterministic seed 路径、return-logprob 路径，或者 Talker 周边模型图。

## 步骤 6：从模型 config 读取 Qwen3-Omni Talker 真实采样形状

目的：
- 不加载权重，仅从 config 读取 Talker 参数，让 sampler 复现更接近真实 Qwen3-Omni Talker 路径。

命令概要：
- 只加载 config：
  `AutoConfig.from_pretrained("/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True)`
- 打印 Talker 相关字段。

观察结果：
- 顶层 config：`Qwen3OmniMoeConfig`
- Talker config：`Qwen3OmniMoeTalkerConfig`
- `talker_config.codec_eos_token_id`：`2150`
- `talker_config.num_code_groups`：`16`
- `talker_config.text_config.vocab_size`：`3072`
- `talker_config.text_config.hidden_size`：`1024`

遇到的问题：
- Transformers warning：
  `Unrecognized keys in rope_parameters for 'rope_type'='default': {'mrope_section', 'interleaved'}`

解决方式：
- 无需处理。这个 warning 不影响读取 Talker 相关字段。

## 步骤 7：使用 Qwen3-Omni Talker 真实 vocab 复现 SGLang Sampler

目的：
- 使用真实 Talker vocab size `3072` 和 BF16/FP32 dtype 组合测试 sampler。
- 专门确认 deterministic seed sampling 是否是失败触发点。
- 检查 Ascend fused sampler 是否有 dtype 约束。

命令概要：
- 使用：
  `logits = torch.randn(batch, 3072, device="npu:0", dtype=...)`
- 使用接近 Qwen 默认的 top-k/top-p 参数：
  - `temperature = 0.9`
  - `top_k = 50`
  - `top_p = 1.0`
- 测试：
  - `sampling_backend="pytorch"`，FP32 logits，带 `sampling_seed`
  - `sampling_backend="pytorch"`，FP32 logits，`sampling_seed=None`
  - `sampling_backend="pytorch"`，BF16 logits，带 `sampling_seed`
  - `sampling_backend="pytorch"`，BF16 logits，`sampling_seed=None`
  - `sampling_backend="ascend"`，BF16 logits + FP32 `top_ps/min_ps`
  - `sampling_backend="ascend"`，BF16 logits + BF16 `top_ps/min_ps`

观察结果：
- `pytorch` backend：
  - FP32 + seed：失败。
  - FP32 + no seed：通过。
  - BF16 + seed：失败。
  - BF16 + no seed：通过。
- 带 seed 的 `pytorch` backend 失败栈：
  - `Sampler.forward`
  - `_sample_from_probs`
  - `top_k_top_p_min_p_sampling_from_probs_torch`
  - `multinomial_with_seed`
  - `murmur_hash32`
  - `torch.compile(dynamic=True)`
  - torch_npu Inductor/Triton 路径
  - BiSheng/HIVM 编译失败
- 关键编译错误：
  `fatal error: error in backend: Cannot select: i64 = fp_to_uint`
  出现在 `murmur_hash32_kernel`。
- Python 层最终异常：
  `IndexError: list index out of range`
  来源于 `torch_npu/_inductor/npu_triton_heuristics.py`。
- `ascend` backend：
  - BF16 logits + FP32 `top_ps/min_ps`：失败，报错：
    `Tensor p expected dtype is DT_BFLOAT16 but found DT_FLOAT`
  - BF16 logits + BF16 `top_ps/min_ps`：通过。

遇到的问题：
- 第一次完整 realistic sampler sweep 输出很长，被终端截断。

解决方式：
- 重跑了更聚焦的 6 个 case，并使用简短 case label。

当前判断：
- 已经在不加载 Qwen3-Omni 的情况下复现了一个明确的小张量根因：
  SGLang 的 seeded PyTorch sampler 路径与当前 torch_npu/CANN 编译栈不兼容，因为 `multinomial_with_seed -> murmur_hash32 -> torch.compile` 会生成 BiSheng 无法编译的 kernel。
- Qwen3-Omni Talker 路径高度可疑，因为 request builder 会在请求没有 seed 时自动补 seed，后续 static sampling info 又会把 `sampling_seed` 传给 SGLang sampler。
- 短期规避方向之一：NPU 上禁用 Qwen3-Omni Talker 的 seeded sampling，或者强制走纯 greedy。
- 另一个可能方向：使用 `sampling_backend="ascend"`，但需要保证 BF16 logits 下 `top_ps/min_ps` 等 metadata tensor dtype 满足 fused op 要求。

## 步骤 8：资源释放后的端到端基线复现（进行中）

目的：
- 在不修改远端代码和依赖的前提下，启动 Qwen3-Omni 原始服务并发起 TTS 请求。
- 保留服务启动日志、请求结果和 NPU 使用情况，确认小张量复现的 seed 路径是否就是端到端失败触发点。

计划操作：
- 先检查 `npu-smi info` 和运行中的服务，确认 NPU 已释放且不会影响其他任务。
- 从远端现有脚本和报告中读取已验证的启动参数与 TTS 请求格式。
- 使用模型 `/home/l00951280/weights/Qwen3-Omni-30B-A3B-Instruct` 启动独立端口的服务，日志写入带时间戳的 `/tmp` 文件。
- 发起最小 TTS 请求，采集完整 Python/NPUIR/BiSheng 报错；若服务未就绪或失败，记录实际问题并以最小改动继续定位。

当前状态：
- 用户已确认机器空闲；尚未执行本步骤中的远端命令。
