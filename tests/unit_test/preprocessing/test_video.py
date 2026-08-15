# SPDX-License-Identifier: Apache-2.0
"""Tests for video reader compatibility across qwen-vl-utils versions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.preprocessing.video import (
    _unpack_video_reader_result,
    _video_resize_defaults,
)


def test_unpack_video_reader_result_accepts_legacy_pair() -> None:
    video = torch.zeros((2, 3, 4, 4))

    result, sample_fps = _unpack_video_reader_result((video, 2.5))

    assert result is video
    assert sample_fps == 2.5


def test_unpack_video_reader_result_accepts_metadata_triple() -> None:
    video = torch.zeros((2, 3, 4, 4))
    metadata = {"video_backend": "decord"}

    result, sample_fps = _unpack_video_reader_result((video, metadata, 1.0))

    assert result is video
    assert sample_fps == 1.0


@pytest.mark.parametrize("result", [None, (), (object(), 1.0, {}, 2.0)])
def test_unpack_video_reader_result_rejects_unknown_contract(result) -> None:
    with pytest.raises((TypeError, ValueError), match="Video reader"):
        _unpack_video_reader_result(result)


def test_video_resize_defaults_accepts_legacy_constants() -> None:
    module = SimpleNamespace(
        IMAGE_FACTOR=28,
        VIDEO_MIN_PIXELS=100,
        VIDEO_MAX_PIXELS=200,
        VIDEO_TOTAL_PIXELS=300,
    )

    assert _video_resize_defaults(module) == (28, 100, 200, 300.0)


def test_video_resize_defaults_matches_new_qwen_vl_utils_formula() -> None:
    module = SimpleNamespace(
        SPATIAL_MERGE_SIZE=2,
        VIDEO_MIN_TOKEN_NUM=128,
        VIDEO_MAX_TOKEN_NUM=768,
        MODEL_SEQ_LEN=128000,
    )

    assert _video_resize_defaults(module) == (
        28,
        128 * 28 * 28,
        768 * 28 * 28,
        128000 * 28 * 28 * 0.9,
    )
