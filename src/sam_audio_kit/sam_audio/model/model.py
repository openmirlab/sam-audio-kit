# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torchdiffeq import odeint

from ...core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
from .align import AlignModalities
from .base import BaseModel
from .codec import DACVAE
from .config import SAMAudioConfig
from .text_encoder import T5TextEncoder
from .transformer import DiT
from .vision_encoder import PerceptionEncoder
from ..processor import Batch
from ..ranking import create_ranker

DFLT_ODE_OPT = {"method": "midpoint", "options": {"step_size": 2 / 32}}


class SinusoidalEmbedding(torch.nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        inv_freq = torch.exp(
            -math.log(theta) * torch.arange(half_dim).float() / half_dim
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None):
        if pos is None:
            seq_len, device = x.shape[1], x.device
            pos = torch.arange(seq_len, device=device)

        emb = torch.einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class EmbedAnchors(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, out_dim: int):
        super().__init__()
        self.embed = torch.nn.Embedding(
            num_embeddings + 1, embedding_dim, padding_idx=num_embeddings
        )
        self.gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.proj = torch.nn.Linear(embedding_dim, out_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        if anchor_ids is None:
            return x

        embs = self.embed(anchor_ids.gather(1, anchor_alignment))
        proj = self.proj(embs)
        return x + self.gate.tanh() * proj


@dataclass
class EncodedAudio:
    """Pre-encoded audio features for reuse across multiple separations."""
    audio_features: torch.Tensor       # [B, T, 2C]
    masked_video_features: torch.Tensor  # [B, vision_dim, T]
    anchor_ids: torch.Tensor           # [B, num_anchors]
    anchor_alignment: torch.Tensor     # [B, T]
    audio_pad_mask: torch.Tensor       # [B, T]
    sizes: torch.Tensor                # [B]


@dataclass
class SeparationResult:
    target: torch.Tensor
    residual: torch.Tensor
    noise: torch.Tensor


class SAMAudio(BaseModel):
    config_cls = SAMAudioConfig
    revision = None

    SKIPPABLE = frozenset(
        {"vision_encoder", "visual_ranker", "text_ranker", "span_predictor"}
    )
    """Components that can be left unbuilt with `skip=`; each is None when skipped.

    These are exactly the components `sam_audio_kit.lite` deletes after loading. Building
    them first costs ~20 s and ~7 GB of host RAM on the base model (the span predictor and
    CLAP ranker each pull their own pretrained weights from the Hub) only to be thrown away.
    """

    def __init__(self, cfg: SAMAudioConfig, skip: frozenset = frozenset()):
        super().__init__()
        unknown = set(skip) - self.SKIPPABLE
        if unknown:
            raise ValueError(f"cannot skip {sorted(unknown)}; skippable: {sorted(self.SKIPPABLE)}")
        self.audio_codec = DACVAE(cfg.audio_codec)
        self.text_encoder = T5TextEncoder(cfg.text_encoder)
        self.vision_encoder = (
            None if "vision_encoder" in skip else PerceptionEncoder(cfg.vision_encoder)
        )
        self.transformer = DiT(cfg.transformer)
        self.proj = torch.nn.Linear(cfg.in_channels, cfg.transformer.dim)
        self.align_masked_video = AlignModalities(
            cfg.vision_encoder.dim, cfg.transformer.dim
        )
        self.embed_anchors = EmbedAnchors(
            cfg.num_anchors, cfg.anchor_embedding_dim, cfg.transformer.dim
        )
        self.memory_proj = torch.nn.Linear(cfg.text_encoder.dim, cfg.transformer.dim)
        self.timestep_emb = SinusoidalEmbedding(cfg.transformer.dim)
        self.visual_ranker = (
            None if "visual_ranker" in skip else create_ranker(cfg.visual_ranker)
        )
        self.text_ranker = (
            None if "text_ranker" in skip else create_ranker(cfg.text_ranker)
        )
        if "span_predictor" in skip:
            self.span_predictor = None
            self.span_predictor_transform = None
        elif cfg.span_predictor is not None:
            self.span_predictor = PEAudioFrame.from_config(
                cfg.span_predictor, pretrained=True
            )
            self.span_predictor_transform = PEAudioFrameTransform.from_config(
                cfg.span_predictor
            )

    @property
    def sample_rate(self):
        return self.audio_codec.sample_rate

    def align_inputs(
        self,
        noisy_audio,
        audio_features: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
    ):
        x = torch.cat(
            [
                noisy_audio,
                torch.zeros_like(audio_features),
                audio_features,
            ],
            dim=2,
        )

        projected = self.proj(x)
        aligned = self.align_masked_video(projected, masked_video_features)
        aligned = self.embed_anchors(aligned, anchor_ids, anchor_alignment)
        return aligned

    def forward(
        self,
        noisy_audio: torch.Tensor,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        time: torch.Tensor,
        masked_video_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        anchor_ids: Optional[torch.Tensor] = None,
        anchor_alignment: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the model.  Represents one function evaluation of the ODE.
        In the below descriptions, B is batch size, T is sequence length, C is channel size.
        Note that the size of C and T may vary across arguments (ex. text_features vs. audio_features),
        it is used only to designate a Channel or time/sequence-length dimension respectively.

        Args:
            noisy_audio (torch.Tensor): Noisy audio input tensor (being denoised).
            audio_features (torch.Tensor): Clean audio features [B x T x C].
            text_features (torch.Tensor): Encoded text features tensor [B x T x C].
            time (torch.Tensor): Timestep tensor for positional encoding [B].
            masked_video_features (Optional[torch.Tensor], optional): Masked video features tensor. [B x C x T].
            text_mask (Optional[torch.Tensor], optional): Padding mask for text features. [B x T].
            anchor_ids (Optional[torch.Tensor], optional): Anchor IDs tensor. Defaults to None [B x T].
            anchor_alignment (Optional[torch.Tensor], optional): Anchor alignment tensor. B x T.
            audio_pad_mask (Optional[torch.Tensor], optional): Padding mask for audio input. [B x T].

        Returns:
            torch.Tensor
        """
        aligned_inputs = self.align_inputs(
            noisy_audio,
            audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=anchor_ids,
            anchor_alignment=anchor_alignment,
        )

        memory = timestep_emb = self.timestep_emb(time, pos=time).unsqueeze(1)
        if text_features is not None:
            memory = self.memory_proj(text_features) + timestep_emb

        return self.transformer(
            aligned_inputs,
            time,
            padding_mask=audio_pad_mask,
            memory=memory,
            memory_padding_mask=text_mask,
        )

    def _get_audio_features(self, audios: torch.Tensor):
        audio_features = self.audio_codec(audios).transpose(1, 2)
        return torch.cat([audio_features, audio_features], dim=2)

    def _get_video_features(self, video, audio_features):
        B, T, _ = audio_features.shape
        if video is None:
            dim = self.vision_encoder.dim if self.vision_encoder is not None else self.align_masked_video.conv.weight.shape[1]
            return audio_features.new_zeros(B, dim, T)
        else:
            return self.vision_encoder(video).transpose(1, 2)

    def _repeat_for_reranking(self, tensor, candidates):
        if candidates > 1:
            B = tensor.size(0)
            rest = tensor.shape[1:]
            return (
                tensor.unsqueeze(1)
                .expand(B, candidates, *rest)
                .reshape(B * candidates, *rest)
            )
        else:
            return tensor

    def _unrepeat_from_reranking(self, tensor, candidates):
        return tensor[::candidates]

    def _get_forward_args(self, batch: Batch, candidates: int = 1):
        audio_features = self._get_audio_features(batch.audios)
        text_features, text_mask = self.text_encoder(batch.descriptions)
        masked_video_features = self._get_video_features(
            batch.masked_video, audio_features
        )

        return {
            "audio_features": self._repeat_for_reranking(audio_features, candidates),
            "text_features": self._repeat_for_reranking(text_features, candidates),
            "text_mask": self._repeat_for_reranking(text_mask, candidates),
            "masked_video_features": self._repeat_for_reranking(
                masked_video_features, candidates
            ),
            "anchor_ids": self._repeat_for_reranking(batch.anchor_ids, candidates),
            "anchor_alignment": self._repeat_for_reranking(
                batch.anchor_alignment, candidates
            ),
            "audio_pad_mask": self._repeat_for_reranking(
                batch.audio_pad_mask, candidates
            ),
        }

    def predict_spans(
        self, batch: Batch, audio_features: torch.Tensor, audio_pad_mask: torch.Tensor
    ) -> Batch:
        input = self.span_predictor_transform(text=batch.descriptions).to(
            audio_features.device
        )
        output = self.span_predictor(
            input_features=audio_features[:, :, :128],
            padding_mask=audio_pad_mask,
            return_spans=True,
            **input,
        )
        anchors = [[["+"] + anchor for anchor in anchors] for anchors in output.spans]
        batch.process_anchors(anchors)
        return batch

    @torch.inference_mode()
    def separate(
        self,
        batch: Batch,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
        reranking_candidates: int = 1,
        predict_spans: bool = False,
        return_all_candidates: bool = False,
    ) -> SeparationResult:
        # Encode audio
        forward_args = self._get_forward_args(batch, candidates=reranking_candidates)

        if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
            batch = self.predict_spans(
                batch=batch,
                audio_features=self._unrepeat_from_reranking(
                    forward_args["audio_features"], reranking_candidates
                ),
                audio_pad_mask=self._unrepeat_from_reranking(
                    forward_args["audio_pad_mask"], reranking_candidates
                ),
            )

        audio_features = forward_args["audio_features"]
        B, T, C = audio_features.shape
        C = C // 2  # we stack audio_features, so the actual channels is half

        if noise is None:
            noise = torch.randn_like(audio_features)

        def vector_field(t, noisy_audio):
            res = self.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )
            return res

        states = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=noise.device),
            **ode_opt,
        )
        generated_features = states[-1].transpose(1, 2)
        # generated_features has shape [B, 2C, T].  Reshape to stack along the batch dimension
        wavs = self.audio_codec.decode(generated_features.reshape(2 * B, C, T)).view(
            B, 2, -1
        )

        bsz = wavs.size(0) // reranking_candidates
        sizes = self.audio_codec.feature_idx_to_wav_idx(batch.sizes)
        target_wavs = self.unbatch(
            wavs[:, 0].view(bsz, reranking_candidates, -1), sizes
        )
        residual_wavs = self.unbatch(
            wavs[:, 1].view(bsz, reranking_candidates, -1), sizes
        )

        if return_all_candidates:
            return SeparationResult(
                target=target_wavs,
                residual=residual_wavs,
                noise=noise,
            )

        if (
            reranking_candidates > 1
            and batch.masked_video is not None
            and self.visual_ranker is not None
        ):
            scores = self.visual_ranker(
                extracted_audio=target_wavs,
                videos=batch.masked_video,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        elif reranking_candidates > 1 and self.text_ranker is not None:
            input_audio = [
                audio[:, :size].expand(reranking_candidates, -1)
                for audio, size in zip(batch.audios, sizes, strict=False)
            ]
            scores = self.text_ranker(
                extracted_audio=target_wavs,
                input_audio=input_audio,
                descriptions=batch.descriptions,
                sample_rate=self.audio_codec.sample_rate,
            )
            idxs = scores.argmax(dim=1)
        else:
            idxs = torch.zeros(bsz, dtype=torch.long, device=noise.device)

        return SeparationResult(
            target=[wav[idx] for wav, idx in zip(target_wavs, idxs, strict=False)],
            residual=[
                wavs[idx] for wavs, idx in zip(residual_wavs, idxs, strict=False)
            ],
            noise=noise,
        )

    # --- Differentiable API ---

    def encode_audio(self, batch: Batch) -> EncodedAudio:
        """
        Encode audio features from a batch for reuse across multiple separations.

        The audio encoding uses no_grad internally (DACVAE encoder), so the
        returned features are not differentiable w.r.t. the input audio.
        This is by design — in optimization workflows, the audio is fixed.

        Args:
            batch: Processed batch from SAMAudioProcessor.

        Returns:
            EncodedAudio with all audio-side tensors needed for
            separate_with_embedding().
        """
        with torch.no_grad():
            audio_features = self._get_audio_features(batch.audios)
            masked_video_features = self._get_video_features(
                batch.masked_video, audio_features
            )
        return EncodedAudio(
            audio_features=audio_features,
            masked_video_features=masked_video_features,
            anchor_ids=batch.anchor_ids,
            anchor_alignment=batch.anchor_alignment,
            audio_pad_mask=batch.audio_pad_mask,
            sizes=batch.sizes,
        )

    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text descriptions to T5 embeddings.

        Returns detached tensors suitable for cloning into a nn.Parameter
        for optimization workflows.

        Args:
            texts: List of text descriptions.

        Returns:
            Tuple of (text_features [B, T_text, 768], text_mask [B, T_text]).
            Both are detached from the computation graph.
        """
        with torch.no_grad():
            features, mask = self.text_encoder(texts)
        return features.detach(), mask.detach()

    def separate_with_embedding(
        self,
        encoded_audio: EncodedAudio,
        text_embedding: torch.Tensor,
        text_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        ode_opt: Dict[str, Any] = DFLT_ODE_OPT,
    ) -> torch.Tensor:
        """
        Differentiable separation using pre-encoded audio and a raw text embedding.

        Unlike separate(), this method does NOT use @torch.inference_mode(),
        allowing gradients to flow through:
            text_embedding → memory_proj → DiT → ODE solver → DACVAE decoder → waveform

        Args:
            encoded_audio: Pre-encoded audio features from encode_audio().
            text_embedding: Text embedding tensor [B, T_text, dim].
                Pass a nn.Parameter here for gradient-based optimization.
            text_mask: Text attention mask [B, T_text] from encode_text().
            noise: Fixed noise tensor for determinism across optimization steps.
                If None, random noise is generated (non-deterministic).
            ode_opt: ODE solver options. Defaults to midpoint with step_size=2/32.

        Returns:
            Waveform tensor [B, 2, num_samples] where [:, 0, :] is target
            and [:, 1, :] is residual. Full padded tensor (no trimming)
            to preserve gradient flow.
        """
        audio_features = encoded_audio.audio_features
        B, T, C = audio_features.shape
        C = C // 2

        if noise is None:
            noise = torch.randn_like(audio_features)

        forward_args = {
            "audio_features": audio_features,
            "text_features": text_embedding,
            "text_mask": text_mask,
            "masked_video_features": encoded_audio.masked_video_features,
            "anchor_ids": encoded_audio.anchor_ids,
            "anchor_alignment": encoded_audio.anchor_alignment,
            "audio_pad_mask": encoded_audio.audio_pad_mask,
        }

        def vector_field(t, noisy_audio):
            return self.forward(
                noisy_audio=noisy_audio,
                time=t.expand(noisy_audio.size(0)),
                **forward_args,
            )

        states = odeint(
            vector_field,
            noise,
            torch.tensor([0.0, 1.0], device=noise.device),
            **ode_opt,
        )
        generated_features = states[-1].transpose(1, 2)
        wavs = self.audio_codec.decode(
            generated_features.reshape(2 * B, C, T)
        ).view(B, 2, -1)

        return wavs

    def unbatch(self, wavs: torch.Tensor, sizes: torch.Tensor, time_dim: int = -1):
        result = []
        for row, size in zip(wavs, sizes, strict=False):
            result.append(row.narrow(dim=time_dim, start=0, length=size))
        return result

    def load_state_dict(self, state_dict, strict=True):
        if strict:
            missing_keys, unexpected_keys = super().load_state_dict(
                state_dict, strict=False
            )
            # We load this directly from HF, not in checkpoint
            skip_regex = re.compile(
                "(^text_encoder|^visual_ranker|^text_ranker|^span_predictor)"
            )
            missing_keys = [x for x in missing_keys if not re.search(skip_regex, x)]
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise RuntimeError(
                    f"Missing keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
                )


__all__ = ["SAMAudio", "EncodedAudio", "SeparationResult"]
