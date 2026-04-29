"""
MXFP4 — Microscaling FP4 (OCP MX spec)
========================================
Format  : E2M1  (1 sign + 2 exponent + 1 mantissa bit)
Scaling : shared 8-bit exponent per block of `block_size` elements
Block   : default 32  (OCP MX standard)

Representable values (positive, E2M1, bias=1):
  subnormal : 0.0, 0.5               (exp=00)
  normal    : 1.0, 1.5, 2.0, 3.0,
              4.0, 6.0               (exp=01,10,11)
  → 8 positive + 8 negative + 0 = 15 non-zero + zero

Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

from typing import Tuple
import torch
from .base import BaseQuantizer

# E2M1 lookup table (positive values only, index = 4-bit code & 0x7)
# index:  0     1     2     3     4     5     6     7
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map each element to nearest E2M1 value (positive domain)."""
    lut = _E2M1_VALUES.to(x.device)
    # x: (...,)  →  codes: (...,) in [0,7]
    diff = (x.abs().unsqueeze(-1) - lut).abs()   # (..., 8)
    codes = diff.argmin(dim=-1)                   # (...,)
    return lut[codes] * x.sign()                   # restore sign (sign(0)=0, lut[0]=0 → stays 0)


class MXFP4Quantizer(BaseQuantizer):
    """
    Weight-only MXFP4 quantizer.

    Steps:
      1. Reshape weight into blocks of `block_size`.
      2. Compute per-block shared exponent (max abs → floor(log2)).
      3. Scale each block by 2^(-shared_exp) so values fit E2M1 range.
      4. Round to nearest E2M1 value.
      5. Store codes + shared exponents.
    """

    def __init__(self, block_size: int = 32):
        self.block_size = block_size

    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          q      : quantized fp tensor (same shape, E2M1 grid values * scale)
          scales : per-block shared exponent  (shape: [num_blocks])
        """
        orig_shape = tensor.shape
        # Flatten to 2D: [rows, cols]
        w = tensor.reshape(-1)
        n = w.numel()
        bs = self.block_size

        # Pad to multiple of block_size
        pad = (bs - n % bs) % bs
        if pad:
            w = torch.cat([w, w.new_zeros(pad)])

        w_blocks = w.reshape(-1, bs)                        # [B, bs]
        abs_max = w_blocks.abs().amax(dim=-1, keepdim=True) # [B, 1]

        # Shared exponent: floor(log2(abs_max)), clamped to int8 range
        shared_exp = torch.floor(torch.log2(abs_max.clamp(min=1e-30)))
        shared_exp = shared_exp.clamp(-127, 127).to(torch.int8)  # store as int8

        # Scale blocks to fit E2M1 range [0, 6]
        scale_factor = (2.0 ** shared_exp.float())           # [B, 1]
        w_scaled = w_blocks / scale_factor                   # [B, bs]

        # Round each element to nearest E2M1
        w_q = _nearest_e2m1(w_scaled)                        # [B, bs]

        # Remove padding and restore shape
        w_q = w_q.reshape(-1)[:n].reshape(orig_shape)

        return w_q * scale_factor.reshape(-1).repeat_interleave(bs)[:n].reshape(orig_shape), \
               shared_exp.squeeze(-1)

    def dequantize(self, q: torch.Tensor, shared_exp: torch.Tensor) -> torch.Tensor:
        # q already contains the dequantized float values (scale applied in quantize)
        return q

    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-quantize: returns rounded values at E2M1 grid * shared scale."""
        orig_shape = tensor.shape
        w = tensor.reshape(-1)
        n = w.numel()
        bs = self.block_size

        pad = (bs - n % bs) % bs
        if pad:
            w = torch.cat([w, w.new_zeros(pad)])

        w_blocks = w.reshape(-1, bs)
        abs_max = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)

        shared_exp = torch.floor(torch.log2(abs_max))
        scale_factor = 2.0 ** shared_exp

        w_scaled = w_blocks / scale_factor
        w_q = _nearest_e2m1(w_scaled)
        w_dq = (w_q * scale_factor).reshape(-1)[:n].reshape(orig_shape)
        return w_dq
