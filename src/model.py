"""Whisper Tiny encoder plus an interchangeable classification head.

The architecture is deliberately the same one the reference implementation uses
— Whisper Tiny encoder, shallow classifier, ~8M parameters — so that our numbers
are comparable to a published baseline (8 MB int8 ONNX, ~10-12 ms CPU) instead
of floating free. Where we differ is that the head is swappable and the encoder
window is configurable, because those are the two axes the experiment matrix
sweeps.

Two implementation points worth reading before changing anything:

**Positional-embedding truncation.** ``openai/whisper-tiny`` ships positional
embeddings for 1500 encoder positions, i.e. 30 s of audio. Our window is at most
8 s. Simply feeding a shorter mel spectrogram trips an internal shape assertion,
so :func:`build_backbone` truncates ``embed_positions`` *and* rewrites
``config.max_source_positions`` to agree. Truncating rather than interpolating is
correct here: Whisper's positional embeddings are learned per absolute index, and
the first N indices are exactly the ones that describe the first N frames. The
audio is left-padded so the speech sits at the high indices — which are the
indices we keep only because we keep a prefix of *positions* while the *window*
itself is what got shortened. Both ends stay consistent because the window is
fixed at build time.

**Freezing is the default, unfreezing is an experiment.** ``freeze_encoder=True``
is how E1 through E5 run. E7 unfreezes the top blocks. That ordering is
deliberate: a frozen encoder trains in minutes on a free T4 and gives a clean
read on whether the *head* can separate the classes at all. If it cannot, a
bigger fine-tune is unlikely to rescue it and the problem is upstream in the
data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn

from .features import ENCODER_STRIDE, HOP_LENGTH, N_MELS, encoder_positions_for

HeadKind = Literal["linear", "mlp", "gru", "attn"]
PoolKind = Literal["mean", "last", "max", "attn"]

DEFAULT_BACKBONE = "openai/whisper-tiny"


@dataclass
class ModelConfig:
    """Everything that defines a model variant in the experiment table."""

    backbone: str = DEFAULT_BACKBONE
    window_seconds: float = 8.0
    head: HeadKind = "linear"
    pool: PoolKind = "mean"
    hidden_size: int = 256
    dropout: float = 0.1
    freeze_encoder: bool = True
    unfreeze_top_blocks: int = 0
    n_mels: int = N_MELS
    # Populated at build time so a checkpoint records what it was trained with.
    resolved: dict = field(default_factory=dict)

    @property
    def n_positions(self) -> int:
        return encoder_positions_for(self.window_seconds)

    @property
    def n_mel_frames(self) -> int:
        return self.n_positions * ENCODER_STRIDE

    def to_dict(self) -> dict:
        d = {
            "backbone": self.backbone,
            "window_seconds": self.window_seconds,
            "head": self.head,
            "pool": self.pool,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "freeze_encoder": self.freeze_encoder,
            "unfreeze_top_blocks": self.unfreeze_top_blocks,
            "n_mels": self.n_mels,
        }
        d.update(self.resolved)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {
            k: v
            for k, v in d.items()
            if k in cls.__dataclass_fields__ and k != "resolved"
        }
        return cls(**known)


# --------------------------------------------------------------------------- #
# backbone
# --------------------------------------------------------------------------- #
def build_backbone(cfg: ModelConfig) -> tuple[nn.Module, int]:
    """Load the Whisper encoder, resized to ``cfg.window_seconds``.

    Returns ``(encoder, d_model)``.
    """
    from transformers import WhisperConfig, WhisperModel

    want = cfg.n_positions
    hf_cfg = WhisperConfig.from_pretrained(cfg.backbone)
    if want > hf_cfg.max_source_positions:
        raise ValueError(
            f"window {cfg.window_seconds}s needs {want} encoder positions but "
            f"{cfg.backbone} only provides {hf_cfg.max_source_positions} "
            f"({hf_cfg.max_source_positions * ENCODER_STRIDE * HOP_LENGTH / 16000:g}s). "
            "Shorten the window."
        )

    encoder = WhisperModel.from_pretrained(cfg.backbone).encoder

    if want < encoder.embed_positions.num_embeddings:
        with torch.no_grad():
            kept = encoder.embed_positions.weight[:want].clone()
        encoder.embed_positions = nn.Embedding.from_pretrained(kept, freeze=False)
        # The encoder asserts on input length using this value, so it has to move
        # in lockstep with the embedding table or the forward pass rejects our mel.
        encoder.config.max_source_positions = want

    d_model = int(encoder.config.d_model)
    cfg.resolved.update(
        {
            "d_model": d_model,
            "encoder_layers": int(encoder.config.encoder_layers),
            "n_positions": want,
            "n_mel_frames": cfg.n_mel_frames,
        }
    )
    return encoder, d_model


def apply_freezing(encoder: nn.Module, cfg: ModelConfig) -> None:
    """Freeze the encoder, optionally leaving the top ``n`` blocks trainable.

    The final ``layer_norm`` is unfrozen together with the blocks. Leaving it
    frozen while the blocks beneath it move is a common and quiet mistake: the
    norm's learned scale was fitted to the old activation statistics, so it
    fights the very update you are trying to make.
    """
    if not cfg.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(True)
        return

    for p in encoder.parameters():
        p.requires_grad_(False)

    n = int(cfg.unfreeze_top_blocks)
    if n <= 0:
        return
    layers = encoder.layers
    if n > len(layers):
        raise ValueError(
            f"unfreeze_top_blocks={n} exceeds the encoder's {len(layers)} blocks"
        )
    for block in layers[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)
    if hasattr(encoder, "layer_norm"):
        for p in encoder.layer_norm.parameters():
            p.requires_grad_(True)


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
class AttentionPool(nn.Module):
    """Single-query attention pooling over encoder positions.

    Included because mean pooling dilutes the signal: the endpoint evidence lives
    in the last few hundred milliseconds, and averaging it against 7 s of
    mid-utterance frames shrinks it by an order of magnitude. Attention pooling
    lets the model decide where to look, at the cost of ``d_model`` parameters.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, D)
        w = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.einsum("bt,btd->bd", w, x)


def _pool(kind: PoolKind, d_model: int) -> nn.Module:
    if kind == "mean":
        return _Lambda(lambda x: x.mean(dim=1))
    if kind == "max":
        return _Lambda(lambda x: x.amax(dim=1))
    if kind == "last":
        # Audio is left-padded, so the final position is always real speech —
        # this is only meaningful because of that padding choice.
        return _Lambda(lambda x: x[:, -1, :])
    if kind == "attn":
        return AttentionPool(d_model)
    raise ValueError(f"unknown pool: {kind!r}")


class _Lambda(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    """Pool then one linear layer. The E1 head, and the reference design."""

    def __init__(self, d_model: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.pool = _pool(cfg.pool, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(self.pool(x)))).squeeze(-1)


class MLPHead(nn.Module):
    """Pool then two layers with a GELU between. E3."""

    def __init__(self, d_model: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.pool = _pool(cfg.pool, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, cfg.hidden_size),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.pool(x)).squeeze(-1)


class GRUHead(nn.Module):
    """Bidirectional GRU over encoder frames, then classify. E4.

    Hypothesis being tested: endpointing is about *trajectory* — falling pitch,
    decaying energy, a filler that trails rather than resolves — and a pooled
    representation throws the ordering away. A recurrent head keeps it.

    Bidirectional is defensible here despite this being a streaming problem: the
    detector runs on a completed window, not on an unbounded future, so the
    backward pass only ever sees audio that has already arrived.
    """

    def __init__(self, d_model: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=cfg.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(self.norm(x))
        # Last forward state and last backward state, which for a bidirectional
        # GRU sit at opposite ends of the sequence.
        fwd = out[:, -1, : self.gru.hidden_size]
        bwd = out[:, 0, self.gru.hidden_size :]
        return self.fc(self.drop(torch.cat([fwd, bwd], dim=-1))).squeeze(-1)


class AttnHead(nn.Module):
    """Attention pooling then linear. Isolates pooling from head capacity."""

    def __init__(self, d_model: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.pool = AttentionPool(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(self.pool(x)))).squeeze(-1)


_HEADS = {"linear": LinearHead, "mlp": MLPHead, "gru": GRUHead, "attn": AttnHead}


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class TurnDetector(nn.Module):
    """Log-mel in, one logit out. Positive logit means "turn has ended"."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder, d_model = build_backbone(cfg)
        apply_freezing(self.encoder, cfg)
        if cfg.head not in _HEADS:
            raise ValueError(f"unknown head {cfg.head!r}; choose from {sorted(_HEADS)}")
        self.head = _HEADS[cfg.head](d_model, cfg)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """``(B, n_mels, n_frames)`` → ``(B,)`` logits."""
        if mel.dim() != 3:
            raise ValueError(f"expected (B, n_mels, n_frames), got {tuple(mel.shape)}")
        hidden = self.encoder(mel).last_hidden_state  # (B, T, D)
        return self.head(hidden)

    @torch.no_grad()
    def probabilities(self, mel: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(mel))

    # -- bookkeeping the experiment table needs ----------------------------- #
    def parameter_counts(self) -> dict[str, int]:
        enc = sum(p.numel() for p in self.encoder.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "encoder": enc,
            "head": head,
            "total": enc + head,
            "trainable": trainable,
        }

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        c = self.parameter_counts()
        return (
            f"{self.cfg.backbone} + {self.cfg.head} head "
            f"(pool={self.cfg.pool}, window={self.cfg.window_seconds:g}s) | "
            f"{c['total'] / 1e6:.2f}M params, {_fmt_count(c['trainable'])} trainable"
        )


def _fmt_count(n: int) -> str:
    """Human-readable parameter count.

    A frozen-encoder run trains ~1.2k parameters; rendering that as "0.00M"
    reads as a bug in the freezing rather than as the point of the experiment.
    """
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def build_model(cfg: ModelConfig | dict) -> TurnDetector:
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return TurnDetector(cfg)
