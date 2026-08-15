# SPDX-License-Identifier: Apache-2.0
"""Portable operator contracts used by the Ming AudioVAE."""

from __future__ import annotations

import torch

from sglang_omni.models.ming_omni.talker.audio_vae.istft import ISTFT


def test_ming_sized_istft_is_finite_on_cpu() -> None:
    module = ISTFT(n_fft=3528, hop_length=882, win_length=3528)
    real = torch.randn(1, 1765, 8)
    imaginary = torch.randn_like(real)
    spectrogram = torch.complex(real, imaginary)

    waveform, audio_buffer, window_buffer = module(spectrogram)

    assert waveform.shape == (1, 7056)
    assert torch.isfinite(waveform).all()
    assert audio_buffer is None
    assert window_buffer is None
