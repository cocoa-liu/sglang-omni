# Ascend container images

The NPU Dockerfile installs the checked-out Omni source on a pinned SGLang
0.5.19 / CANN 9.0.0 stack. The image includes Python dependencies, FFmpeg and
SoundFile, but not model weights, host drivers or TorchCodec. Optional model
dependencies remain documented per model; an image build is not a claim that
all Omni models work on NPU.

## Build

Run from the repository root on ARM64 (or with an ARM64 build platform):

```bash
# A3, the default pinned base
docker build -f docker/npu.Dockerfile -t sglang-omni:npu-a3 .

# A2 / 910B, the matching pinned base
docker build -f docker/npu.Dockerfile \
  --build-arg SGLANG_IMAGE=lmsysorg/sglang:v0.5.19-cann9.0.0-910b@sha256:19beb175fe8b5a636a14c0f8a95aa92686a36e09b826f214be476b2ae22e7188 \
  -t sglang-omni:npu-910b .
```

Use the image matching both CPU architecture and NPU family. An A2 build
does not establish A3 binary compatibility, or vice versa. The host must supply
compatible Ascend drivers/firmware and device access. Follow the NPU installation
guide for host prerequisites, then the chosen model's cookbook for its model
directory, configuration and API requests. Do not reinstall Omni's CUDA extras
inside this image.

## Publishing

Both workflows build ARM64 A3 and 910B variants with independent caches:

- Release tags: `vVERSION-cann9.0.0-a3` and `vVERSION-cann9.0.0-910b`.
- Source tags: `sha-SHA-cann9.0.0-DEVICE`.
- Nightly rolling tags: `main-cann9.0.0-a3` and `main-cann9.0.0-910b`.

Nightly is scheduled for 14:00 UTC (22:00 Beijing). Scheduled builds are disabled
until repository variable `NPU_DOCKER_PUBLISH_ENABLED=true` is configured. Manual
and pull-request builds can be validated without publishing credentials.

Maintainers must first confirm the destination repository and grant a token
push access. Configure `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets and,
if needed, `NPU_DOCKER_IMAGE` (default `lmsysorg/sglang-omni`). Only the upstream
repository can publish; PR events never log in or push. The manual release
workflow additionally requires its `publish` input. A successful build without
these settings is not a successful publication.

The Dockerfile performs package import checks; GitHub-hosted build runners do
not perform NPU inference. Validate model startup and requests on the matching
hardware before promoting an image for users. Pin the tested registry digest
for deployments instead of relying on a rolling tag.
