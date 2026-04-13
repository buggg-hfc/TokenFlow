"""
HiF4 — Hierarchical FP4 Quantization
======================================
A research-oriented FP4 scheme that adds a *hierarchical* two-pass scaling:

  Pass 1 – column-wise (output-feature) scale
            Captures systematic magnitude differences between output neurons.
  Pass 2 – block-wise  (within each column) scale
            Fine-grained per-block shared exponent, same as MXFP4.

The hierarchical structure improves accuracy when weights have high
inter-column variance (e.g., after layer norm, attention projections).

Format  : E2M1 (same 4-bit grid as MXFP4 / NVFP4)
Scaling : col_scales  [out_features]      – fp32, per output neuron
          blk_scales  [out_features * B]  – int8, per block within column
Block   : default 32

Reference: "HiF4: Hierarchical FP4 Quantization for LLMs" (internal / draft)
"""

from typing import Tuple
import torch
from .base import BaseQuantizer

_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    lut = _E2M1_VALUES.to(x.device)
    diff = (x.abs().unsqueeze(-1) - lut).abs()
    codes = diff.argmin(dim=-1)
    return lut[codes] * x.sign().clamp(min=0)


class HiF4Quantizer(BaseQuantizer):
    """
    Hierarchical FP4 quantizer with column + block scaling.

    Weight tensor assumed shape: [out_features, in_features]
    (standard Linear weight layout in PyTorch).

    Steps:
      1. Reshape to 2-D [out, in].
      2. Per-column scale: max|w[:,j]| / 6 for each output neuron j.
      3. Divide each column by its scale.
      4. Within each column, apply MXFP4-style block shared exponents.
      5. Store col_scales (fp32) and blk_scales (int8).
    """

    def __init__(self, block_size: int = 32):
        self.block_size = block_size

    # ------------------------------------------------------------------
    def quantize(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          q          : dequantized fp tensor (same shape as input)
          col_scales : [out_features] fp32
          blk_scales : [out_features * num_blocks_per_col] int8
        """
        orig_shape = tensor.shape
        # Work in 2-D regardless of original shape
        if tensor.dim() == 1:
            w2d = tensor.unsqueeze(0)       # treat as single row
        else:
            w2d = tensor.reshape(tensor.shape[0], -1)   # [out, in*...]

        out_dim, in_dim = w2d.shape
        bs = self.block_size

        # ---- Pass 1: column-wise scale (per output neuron) ----
        col_max = w2d.abs().amax(dim=1).clamp(min=1e-30)  # [out]
        col_scales = (col_max / 6.0).to(torch.float32)    # [out]
        w_col_norm = w2d / col_scales.unsqueeze(1)         # [out, in]

        # ---- Pass 2: block-wise shared exponent ----
        pad = (bs - in_dim % bs) % bs
        if pad:
            w_col_norm = torch.cat(
                [w_col_norm, w_col_norm.new_zeros(out_dim, pad)], dim=1
            )
        num_blocks_per_col = (in_dim + pad) // bs         # B

        # [out, B, bs]
        w_blocks = w_col_norm.reshape(out_dim, num_blocks_per_col, bs)

        blk_max = w_blocks.abs().amax(dim=-1).clamp(min=1e-30)  # [out, B]
        blk_exp = torch.floor(torch.log2(blk_max))              # [out, B]
        blk_exp_i8 = blk_exp.clamp(-127, 127).to(torch.int8)    # [out, B]

        blk_scale = (2.0 ** blk_exp.float()).unsqueeze(-1)      # [out, B, 1]
        w_scaled = w_blocks / blk_scale                          # [out, B, bs]

        w_q = _nearest_e2m1(w_scaled)                            # [out, B, bs]

        # Dequantize
        w_dq = (w_q * blk_scale).reshape(out_dim, -1)[:, :in_dim]  # [out, in]
        w_dq = w_dq * col_scales.unsqueeze(1)                       # restore col scale
        w_dq = w_dq.reshape(orig_shape)

        return w_dq, col_scales, blk_exp_i8.reshape(-1)

    # ------------------------------------------------------------------
    def dequantize(
        self,
        q: torch.Tensor,
        col_scales: torch.Tensor,
        blk_scales: torch.Tensor,
    ) -> torch.Tensor:
        # q already contains the fully dequantized float values
        return q

    # ------------------------------------------------------------------
    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-quantize (training): hierarchical quant noise."""
        orig_shape = tensor.shape

        if tensor.dim() == 1:
            w2d = tensor.unsqueeze(0)
        else:
            w2d = tensor.reshape(tensor.shape[0], -1)

        out_dim, in_dim = w2d.shape
        bs = self.block_size

        col_max = w2d.abs().amax(dim=1).clamp(min=1e-30)
        col_scales = col_max / 6.0
        w_col_norm = w2d / col_scales.unsqueeze(1)

        pad = (bs - in_dim % bs) % bs
        if pad:
            w_col_norm = torch.cat(
                [w_col_norm, w_col_norm.new_zeros(out_dim, pad)], dim=1
            )
        num_blocks_per_col = (in_dim + pad) // bs

        w_blocks = w_col_norm.reshape(out_dim, num_blocks_per_col, bs)
        blk_max = w_blocks.abs().amax(dim=-1).clamp(min=1e-30)
        blk_exp = torch.floor(torch.log2(blk_max))
        blk_scale = (2.0 ** blk_exp).unsqueeze(-1)

        w_q = _nearest_e2m1(w_blocks / blk_scale)
        w_dq = (w_q * blk_scale).reshape(out_dim, -1)[:, :in_dim]
        w_dq = (w_dq * col_scales.unsqueeze(1)).reshape(orig_shape)
        return w_dq
