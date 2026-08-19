# SPDX-License-Identifier: Apache-2.0
"""Audio encoder component for Qwen3-Omni."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.common import (
    assert_module_device,
    assert_tensor_tree_device,
    load_thinker_config,
    resolve_component_device,
)
from sglang_omni.models.weight_loader import load_module, resolve_dtype
from sglang_omni.utils import instantiate_module

AUDIO_TOWER_PREFIX = ("thinker.audio_tower.", "audio_tower.")
AUDIO_TOWER_CLASS = hf_modeling.Qwen3OmniMoeAudioEncoder


def _build_audio_tower(
    model_path: str,
    *,
    thinker_cfg: object,
    torch_dtype: torch.dtype | None,
    device: str,
) -> nn.Module:
    audio_cfg = thinker_cfg.audio_config
    audio_tower = instantiate_module(AUDIO_TOWER_CLASS, audio_cfg)
    return load_module(
        audio_tower,
        model_path,
        prefix=AUDIO_TOWER_PREFIX,
        dtype=torch_dtype,
        device=device,
        strict=True,
    )


class Qwen3OmniAudioEncoder(nn.Module):
    """Audio tower extracted from the HF thinker."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = None,
        dtype: str | torch.dtype | None = None,
    ) -> None:
        super().__init__()
        device = resolve_component_device(
            device=device,
            gpu_id=None,
            component="qwen3_omni_audio_encoder",
        )
        torch_dtype = resolve_dtype(dtype)
        thinker_cfg = load_thinker_config(model_path)
        self._device = torch.device(device)
        self.audio_tower = _build_audio_tower(
            model_path,
            thinker_cfg=thinker_cfg,
            torch_dtype=torch_dtype,
            device=device,
        )
        assert_module_device(
            self.audio_tower,
            self._device,
            component="qwen3_omni_audio_encoder",
        )
        self._forward_device_verified = False
        self._downsample_lengths = hf_modeling._get_feat_extract_output_lengths

    def forward(
        self,
        *,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = (
                input_features.permute(0, 2, 1)[feature_attention_mask.bool()]
                .permute(1, 0)
                .contiguous()
            )
        if audio_feature_lengths is None:
            raise ValueError(
                "audio_feature_lengths or feature_attention_mask is required"
            )

        audio_feature_lengths = audio_feature_lengths.to(self._device, dtype=torch.long)
        input_features = input_features.to(
            device=self._device, dtype=self.audio_tower.dtype
        )
        if not self._forward_device_verified:
            assert_tensor_tree_device(
                {
                    "input_features": input_features,
                    "audio_feature_lengths": audio_feature_lengths,
                },
                self._device,
                component="qwen3_omni_audio_encoder",
                boundary="forward_input",
            )
        outputs = self.audio_tower(input_features, feature_lens=audio_feature_lengths)
        audio_embeds = outputs.last_hidden_state
        audio_output_lengths = self._downsample_lengths(audio_feature_lengths)
        if not self._forward_device_verified:
            assert_tensor_tree_device(
                {
                    "audio_embeds": audio_embeds,
                    "audio_output_lengths": audio_output_lengths,
                },
                self._device,
                component="qwen3_omni_audio_encoder",
                boundary="forward_output",
            )
            logging.getLogger(__name__).info(
                "component_forward_device_verified component=%s device=%s",
                "qwen3_omni_audio_encoder",
                self._device,
            )
            self._forward_device_verified = True
        return {
            "audio_embeds": audio_embeds,
            "audio_feature_lengths": audio_feature_lengths,
            "audio_output_lengths": audio_output_lengths,
        }
