#!/usr/bin/env bash
# Run only in a trusted checkout on an explicitly reserved NPU.
set -euo pipefail

: "${NPU_CI_IMAGE:?Set NPU_CI_IMAGE to a compatible Omni environment image}"
: "${NPU_CI_DEVICE:?Set NPU_CI_DEVICE to one reserved physical NPU ID}"
[[ "$NPU_CI_DEVICE" =~ ^[0-9]+$ ]] || {
  echo 'NPU_CI_DEVICE must be one physical NPU ID.' >&2
  exit 2
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
results_dir="${NPU_CI_RESULTS_DIR:-${repo_root}/npu-ci-results}"
mkdir -p "$results_dir"
results_dir="$(cd "$results_dir" && pwd)"
run_dir="$(mktemp -d "$results_dir/run-XXXXXX")"
container_name="omni-npu-ci-$(basename "$run_dir")-$$"

# Invoked indirectly by the EXIT trap.
# shellcheck disable=SC2317
cleanup() {
  local status=$?
  trap - EXIT
  if docker inspect "$container_name" >/dev/null 2>&1; then
    docker logs "$container_name" > "$run_dir/container.log" 2>&1 || true
    docker cp "$container_name:/results/." "$run_dir/" || true
    # Remove this run's container only; never kill other users' NPU processes.
    docker rm -f "$container_name" >/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

device_args=()
for device in "/dev/davinci${NPU_CI_DEVICE}" /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc; do
  [[ -e "$device" ]] || { echo "Missing device: $device" >&2; exit 2; }
  device_args+=(--device "$device")
done
mount_args=()
for path in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/sbin /etc/ascend_install.info; do
  [[ -e "$path" ]] || { echo "Missing runtime path: $path" >&2; exit 2; }
  mount_args+=(-v "$path:$path:ro")
done
if [[ -d /var/queue_schedule ]]; then
  mount_args+=(-v /var/queue_schedule:/var/queue_schedule)
fi

if [[ $# -eq 0 ]]; then
  set -- tests/test_ci/test_npu_runtime.py
fi
git -C "$repo_root" rev-parse HEAD > "$run_dir/source-sha.txt"
git -C "$repo_root" status --short > "$run_dir/source-status.txt"
docker image inspect "$NPU_CI_IMAGE" > "$run_dir/image.json" 2>/dev/null || {
  docker pull "$NPU_CI_IMAGE"
  docker image inspect "$NPU_CI_IMAGE" > "$run_dir/image.json"
}
# Privileged is required by some Ascend deployments. Use a dedicated trusted
# runner: device visibility is not a security boundary with privileged Docker.
docker create --name "$container_name" --privileged --network host --shm-size 8g \
  "${device_args[@]}" "${mount_args[@]}" \
  -v "$repo_root:/checkout:ro" \
  -e "ASCEND_RT_VISIBLE_DEVICES=$NPU_CI_DEVICE" \
  -e OMP_NUM_THREADS=8 -e OMNI_RUN_NPU_TESTS=1 \
  "$NPU_CI_IMAGE" bash -c '
    set -eo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    mkdir -p /results /tmp/omni-ci-source
    tar -C /checkout --exclude=.git --exclude=npu-ci-results --exclude=.venv \
      --exclude=__pycache__ -cf - . | tar -C /tmp/omni-ci-source -xf -
    cd /tmp/omni-ci-source
    cp pyproject_npu.toml pyproject.toml
    python -m pip install --no-deps --no-build-isolation -e .
    python -m pip install --no-deps pytest==8.3.5
    export PYTHONPATH=/tmp/omni-ci-source
    python -m pip freeze > /results/packages.txt
    npu-smi info > /results/npu-smi.txt
    python -m pytest "$@" -v -s --junitxml=/results/junit.xml
  ' bash "$@" > "$run_dir/container-id.txt"
docker start -a "$container_name"
exit "$(docker inspect --format '{{.State.ExitCode}}' "$container_name")"
