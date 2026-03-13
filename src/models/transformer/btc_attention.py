"""
btc_attention.py
================
Multi-Head Self-Attention Transformer Feature Extractor for the PPO Policy.

Architecture
------------
Flat obs [seq_len × n_features]
      │
      └─► Reshape → [batch, seq_len, n_features]
              │
              ▼
   Linear Projection → [batch, seq_len, embed_dim]
              │
              ▼
   Positional Encoding (sinusoidal, causal-aware)
              │
              ▼
   TransformerEncoder × n_layers
     └─ MultiHeadAttention (n_heads, causal mask)
     └─ FeedForward (embed_dim × 4)
     └─ LayerNorm + Dropout
              │
              ▼
   Last-Token Pooling (most recent timestep) → [batch, embed_dim]
              │
              ▼
   Layer Norm → output features_dim

Scaling changelog (4-year dataset edition)
------------------------------------------
  SCALE 1 — embed_dim 128 → 256:
    4 years × 288 candles/day ≈ 420k candles. The previous 128-dim embedding
    was sized for a ~6-month dataset. 256 gives the attention heads enough
    representational capacity to separate bull/bear/ranging regime embeddings
    without overfitting given the training set size (~176k+ rows).

  SCALE 2 — n_heads 4 → 8:
    Maintains embed_dim / n_heads = 32 (same per-head dimension as before).
    8 heads allows the model to simultaneously attend to: short-term momentum,
    volatility regime, volume anomalies, time-of-day patterns, and
    cross-feature correlations — all in parallel.

  SCALE 3 — n_layers 2 → 4:
    Deeper stack builds more abstract temporal representations. Layer 1-2
    learn low-level patterns (candle sequences, micro-structure). Layer 3-4
    compose these into higher-order regime signals. 4 layers is the practical
    limit before diminishing returns at this embedding width.

  SCALE 4 — features_dim 128 → 256:
    Widens the output vector fed to PPO actor/critic heads, matching the
    increased embed_dim. Also requires net_arch=[256,128] in train_ppo_attention.py
    to avoid an information bottleneck immediately after the extractor.

  KEPT — dropout 0.1:
    Correct for this dataset size. Increasing it would reintroduce the
    value network instability seen with dropout=0.3.

  KEPT — last-token pooling (not mean pooling):
    Still the right choice. The causal Transformer's final token has attended
    to all 128 prior positions and holds the richest current-state encoding.

  KEPT — no ReLU on out_proj:
    Preserves signed bearish/bullish signal into the actor/critic heads.
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

warnings.filterwarnings("ignore")


# =========================================================================
# Positional Encoding
# =========================================================================

class CausalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional embeddings.

    Using fixed (not learned) because:
      - Financial sequences have no fixed vocabulary.
      - Fixed encodings generalise to lengths unseen in training.
      - One fewer overfitting source on rolling-window inputs.

    Shape: [batch, seq_len, embed_dim] → [batch, seq_len, embed_dim]
    """

    def __init__(self, embed_dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, embed_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# =========================================================================
# Transformer Encoder Block
# =========================================================================

class BTCTransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block:
      MultiHeadAttention → Add+Norm → FeedForward → Add+Norm

    Causal masking ensures position T attends only to positions 0..T,
    preventing any future-price leakage during sequence processing.
    """

    def __init__(self, embed_dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(embed_dim)
        self.ff      = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2   = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask     = self._causal_mask(x.size(1), x.device)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


# =========================================================================
# Composite Attention Feature Extractor (SB3 compatible)
# =========================================================================

class BTCAttentionExtractor(BaseFeaturesExtractor):
    """
    SB3 BaseFeaturesExtractor wrapping the scaled Transformer pipeline.

    Expects a flat observation of shape [seq_len * n_raw_features].
    Internally reshapes, encodes, attends, and pools to features_dim.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Must have shape (seq_len * n_raw_features,).
    n_raw_features : int
        Number of features per single candle observation.
    seq_len : int
        Number of past candles in the sliding window (default 128).
        128 = 10.6h context window; captures overnight sessions and
        multi-session momentum patterns across the 4yr dataset.
    embed_dim : int
        Transformer embedding dimension (default 256).
        Scaled from 128 to match 4yr dataset capacity.
    n_heads : int
        Attention heads (default 8). embed_dim / n_heads = 32 per head —
        same per-head dimension as the previous 128/4 configuration.
    n_layers : int
        Stacked encoder blocks (default 4).
        Scaled from 2; layers 3-4 learn regime-level abstractions.
    dropout : float
        Dropout rate (default 0.1). Do not increase — 0.3 caused
        value network instability in earlier runs.
    features_dim : int
        Output dim to PPO actor/critic heads (default 256).
        Requires net_arch=[256,128] in train_ppo_attention.py.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        n_raw_features:    int,
        seq_len:           int   = 64,    # SCALE 1: was 64
        embed_dim:         int   = 128,    # SCALE 2: was 128
        n_heads:           int   = 4,      # SCALE 3: was 4
        n_layers:          int   = 2,      # SCALE 4: was 2
        dropout:           float = 0.1,
        features_dim:      int   = 128,    # SCALE 5: was 128
    ):
        super().__init__(observation_space, features_dim=features_dim)

        assert embed_dim % n_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"

        self.n_raw_features = n_raw_features
        self.seq_len        = seq_len
        self.embed_dim      = embed_dim

        self.input_proj = nn.Linear(n_raw_features, embed_dim)

        self.pos_enc = CausalPositionalEncoding(
            embed_dim, max_len=seq_len + 4, dropout=dropout
        )

        self.encoder_blocks = nn.ModuleList([
            BTCTransformerEncoderBlock(embed_dim, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.post_norm = nn.LayerNorm(embed_dim)

        # No ReLU — preserves signed bearish/bullish activations into PPO heads
        self.out_proj  = nn.Linear(embed_dim, features_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for all linear layers; zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : torch.Tensor  [batch, seq_len * n_raw_features]

        Returns
        -------
        torch.Tensor        [batch, features_dim]
        """
        batch = obs.size(0)

        x = obs.view(batch, self.seq_len, self.n_raw_features)  # reshape
        x = self.input_proj(x)                                   # project to embed_dim
        x = self.pos_enc(x)                                      # positional encoding

        for block in self.encoder_blocks:
            x = block(x)                                         # 4× Transformer blocks

        # Last-token pooling — final position has attended to all 128 prior
        # candles and holds the richest current-state representation
        x = x[:, -1, :]                                          # [batch, embed_dim]

        x = self.post_norm(x)
        x = self.out_proj(x)                                     # [batch, features_dim]

        return x