# Quickstart — Ascend NPU

Run SGLang-Omni in an ARM64 Ascend container. The image includes Omni source,
SGLang 0.5.19, CANN 9.0.0, PyTorch / `torch_npu` 2.10 and common audio dependencies.
Model weights, host drivers and TorchCodec are not included. Model support on
NPU and optional dependencies must be checked separately; the image does not
enable every Omni model automatically.

## Prerequisites

- An ARM64 Linux host with Ascend A3 or A2 / 910B devices and Docker.
- Compatible host drivers and firmware; see the HDK links in
  [NPU prerequisites](installation_npu.md#prerequisites). Check `npu-smi info`
  on the host before starting.
- Enough device memory and disk space for the selected model. The Ming example
  below uses four devices; adjust placement and memory settings for your hardware.

The runtime is already installed inside the image. Do not rerun the source
installation or TorchCodec stack-upgrade scripts inside it.

## 1. Prepare the image

Until official images are published, build from the repository root. Choose
**one** build matching your hardware; A2 and A3 operator binaries are not
interchangeable.

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# A3
IMAGE=sglang-omni:npu-a3
docker build -f docker/npu.Dockerfile -t "$IMAGE" .
```

For A2 / 910B, replace the last two commands with:

```bash
IMAGE=sglang-omni:npu-910b
docker build -f docker/npu.Dockerfile \
  --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.19-cann9.0.0-910b@sha256:19beb175fe8b5a636a14c0f8a95aa92686a36e09b826f214be476b2ae22e7188 \
  -t "$IMAGE" .
```

If your operator provides a published image, set `IMAGE` to its matching
registry reference and run `docker pull "$IMAGE"` instead. Use a tested digest
for deployments. A successful build alone does not verify inference on A2 or A3.

## 2. Prepare the model

This example uses the main-branch Ming-Omni text pipeline. Download weights to
a host directory using the image's Hugging Face CLI, or use an existing checkpoint:

```bash
MODEL_DIR=/mnt/models/Ming-flash-omni-2.0
mkdir -p "$MODEL_DIR"
docker run --rm \
  -v "$MODEL_DIR:/model" \
  "$IMAGE" hf download inclusionAI/Ming-flash-omni-2.0 --local-dir /model

# Ming uses the Preview checkpoint's tokenizer files.
docker run --rm \
  -v "$MODEL_DIR:/model" \
  "$IMAGE" hf download inclusionAI/Ming-flash-omni-Preview \
  tokenizer.json tokenizer_config.json special_tokens_map.json --local-dir /model
```

## 3. Start the service

The example exposes physical devices 0–3 and binds the API to host loopback.
Replace the device IDs and driver mount paths if your host differs. `--privileged`
grants broad host access: use this quickstart only on a trusted host; production
deployments should use their platform's restricted device-access configuration.

```bash
docker run -d --name omni-npu \
  --privileged --shm-size 32g \
  -p 127.0.0.1:8000:8000 \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$MODEL_DIR:/model:ro" \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  "$IMAGE" bash -lc '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    exec sgl-omni serve \
      --model-path /model --model-name ming-omni \
      --text-only --thinker.tp_size 4 --thinker.gpu "[0,1,2,3]" \
      --host 0.0.0.0 --port 8000
  '

docker logs -f omni-npu
```

The CLI retains the name `gpu` for stage placement on NPU; these are logical
indices within the visible devices. Text-only mode skips the talker. Wait for
startup to complete, then check the API from another host terminal:

```bash
curl --fail http://localhost:8000/v1/models
```

## 4. Send a request

Replace the message with your own prompt. The response text is in
`choices[0].message.content`.

```bash
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ming-omni",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    "modalities": ["text"],
    "max_tokens": 128,
    "temperature": 0
  }'
```

See the [Ming-Omni cookbook](../cookbook/ming_omni.md) for multimodal requests
and speech configuration. Its CUDA device-selection examples must be adapted
to NPU placement. Other models require their own supported NPU configuration;
changing only the model directory is not sufficient.

Stop and remove the container when finished; downloaded host weights remain:

```bash
docker stop omni-npu
docker rm omni-npu
```
