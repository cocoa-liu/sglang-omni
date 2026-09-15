# NPU CI bring-up

The workflow is manual-only and disabled until `NPU_CI_ENABLED=true` is set as
a repository variable. It does not run on pull requests or a schedule and is
not a required check. A skipped job is not hardware validation.

## Runner prerequisites

- A trusted, dedicated ARM64 Linux runner labeled `omni-npu-a3` or
  `omni-npu-910b`, with Docker and matching Ascend driver/firmware installed.
- Set repository variable `NPU_CI_DEVICE` to one reserved physical device ID.
  Both hardware pools must expose that ID, or use separate repositories/environments
  when their allocation differs. Do not pick a busy device automatically.
- Provide a compatible Omni image pinned by digest when dispatching. The image
  is an environment, not the code under test: the checked-out source is copied
  and installed inside a disposable container without replacing its NPU stack.
- The runner needs registry and PyPI access for the image and pytest 8.3.5.
  Model weights are not needed by the runtime checks.

The job uses privileged Docker and host networking, as required by the initial
Ascend deployment. Run trusted code only; do not enable unreviewed fork code on
this runner. The visibility mask is resource selection, not security isolation.
No credentials are passed into the test container.

## Run locally

```bash
NPU_CI_IMAGE=your-registry/omni@sha256:YOUR_DIGEST \
NPU_CI_DEVICE=1 bash scripts/npu/run_ci.sh
```

Logs, package versions, source SHA, image metadata and JUnit results are saved
under `npu-ci-results/run-*`. The script removes only its own container, including
on test failure. If the host/runner is killed outright, an operator must remove
the leftover run container. There is no global NPU process cleanup or automatic
test retry that could hide a reproducible failure.

The smoke suite checks NPU selection, BF16 device computation and NPUGraph
replay. It deliberately fails rather than skips when explicitly enabled on an
unusable NPU. Model behavior and performance are separate regression suites.

## Enable later

1. Provision and reserve hardware, with a matching image for each device family.
2. Run the local command and check cleanup and collected diagnostics.
3. Set the two repository variables and dispatch the workflow on each family.
4. Only after successful runs, add reviewed PR gates or scheduled model tests.

Reference implementations: Omni's `omni-xpu-ci.yaml` and SGLang's
`_npu-single-node-test-stage.yml`. CUDA cleanup scripts and GPU performance
thresholds are intentionally not reused.
