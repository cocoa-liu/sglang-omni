# syntax=docker/dockerfile:1.7
# Base versions and digests are maintained in pyproject_npu.toml.
# The selected digest pins the complete Ascend dependency stack.
# Build from the repository root; source is installed from the build context.
# Pass SGLANG_IMAGE from scripts/npu/config.py --get base-image-a3 (or 910b).
ARG SGLANG_IMAGE
FROM ${SGLANG_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG FFMPEG_VERSION=7:4.4.2-0ubuntu0.22.04.1
ARG LIBSNDFILE_VERSION=1.0.31-2ubuntu0.2
ARG SOX_VERSION=14.4.2+git20190427-2+deb11u2ubuntu0.22.04.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg=${FFMPEG_VERSION} \
        libsndfile1=${LIBSNDFILE_VERSION} \
        sox=${SOX_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Resolve no dependencies here: the lock includes additions to the pinned base.
# In particular, qwen-tts metadata pins a different Transformers version; Omni's
# compatibility layer supports the project's Transformers 5.12.1 instead.
COPY docker/requirements-npu.txt /tmp/requirements-npu.txt
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation -r /tmp/requirements-npu.txt \
    && rm /tmp/requirements-npu.txt

COPY . /workspace/sglang-omni
WORKDIR /workspace/sglang-omni
RUN python scripts/npu/config.py --check-installed \
    && python scripts/npu/config.py --check \
    && cp pyproject_npu.toml pyproject.toml \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && cd / \
    && python -c "import sglang_omni; import librosa; import soundfile" \
    && python -m pip freeze --all > /workspace/python-packages.txt \
    && dpkg-query -W > /workspace/system-packages.txt

# Qwen3-TTS audio input uses Omni's SoundFile fallback when TorchCodec is absent.
# Do not pull an incompatible CUDA-linked TorchCodec wheel into this CPU torch base.
ENTRYPOINT []
CMD ["/bin/bash"]
