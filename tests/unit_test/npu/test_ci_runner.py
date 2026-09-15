# SPDX-License-Identifier: Apache-2.0
"""Validate the opt-in gate without touching Docker or accelerator devices."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/npu/run_ci.sh"


@pytest.mark.parametrize("device", [None, "", "0,1", "-1", "$(touch unsafe)"])
def test_requires_one_explicit_device(device, tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NPU_CI_")}
    env["NPU_CI_IMAGE"] = "test-image"
    if device is not None:
        env["NPU_CI_DEVICE"] = device
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "NPU_CI_DEVICE" in result.stderr
    assert not list(tmp_path.iterdir())


def test_disabled_hardware_tests_collect_without_torch(tmp_path):
    env = dict(os.environ, OMNI_RUN_NPU_TESTS="0")
    result = subprocess.run(
        [
            os.sys.executable,
            "-m",
            "pytest",
            str(SCRIPT.parents[2] / "tests/test_ci/test_npu_runtime.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 skipped" in result.stdout
