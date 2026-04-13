"""
NVFP4 — NVIDIA FP4 (Blackwell / TensorRT-LLM style)
======================================================
Format  : E2M1  (same bit layout as MXFP4)
Scaling : two-level
            L1 – per-tensor  scale  (fp32)
            L2 – per-group   scale  (fp8 E4M3, decoded to fp32 on use)
Group   : default 16  (NVIDIA recommended)

Difference from MXFP4:
  • Smaller group (16 vs 32) → finer-grained L2 scale
  • L2 scales stored as FP8 E4M3 (clipped to ±448) rather than int8 exponent
  • L1 scale normalises the *entire* weight tensor first so L2 codes stay
    in a compact range

Reference: NVIDIA Blackwell Architecture whitepaper / TRT-LLM FP4 docs
"""

from typing import Tuple
import torch
from .base import BaseQuantizer

# ---- shared E2M1 table (same values as MXFP4) ----------------------------
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# FP8 E4M3 max representable value (used to clamp L2 scales before storage)
_FP8_E4M3_MAX = 448.0


def _nearest_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Map each element to nearest E2M1 value (signed)."""
    lut = _E2M1_VALUES.to(x.device)
    diff = (x.abs().unsqueeze(-1) - lut).abs()   # (..., 8)
    codes = diff.argmin(dim=-1)                   # (...,)
    return lut[codes] * x.sign().clamp(min=0)


def _to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate FP8 E4M3 storage by clamping to ±448 and rounding to the
    nearest representable E4M3 value.

    E4M3: bias=7, max_exp=14, max_val=448, subnormals down to ~1.95e-3
    We approximate by clamping + rounding to 8-bit float precision (3 mantissa
    bits → round to nearest 1/8 in the normalised range).
    """
    x = x.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    # round to 3-mantissa-bit precision
    sign = x.sign()
    abs_x = x.abs().clamp(min=1e-30)
    exp = torch.floor(torch.log2(abs_x)).clamp(-9, 8)   # E4M3 exp range
    scale = 2.0 ** exp
    mantissa = (abs_x / scale).clamp(1.0, 1.875)        # [1, 1+7/8]
    mantissa_rounded = (mantissa * 8).round() / 8       # 3-bit mantissa
    return sign * scale * mantissa_rounded


class NVFP4Quantizer(BaseQuantizer):
    """
    Weight-only NVFP4 quantizer (two-level scaling).

    Steps:
      1. Compute L1 (per-tensor) scale = max_abs / 6.0
         so that the largest element maps to ±6, the E2M1 maximum.
      2. Divide the whole tensor by L1 → normalised tensor.
      3. Split into groups of `group_size`.
      4. Compute L2 (per-group) scale = group_max_abs / 6.0.
      5. Clip L2 to FP8 E4M3 range and simulate FP8 storage.
      6. Divide each group by its L2 scale → elements in [-6, 6].
      7. Round to nearest E2M1.
      8. Return dequantized values + (l1_scale, l2_scales_fp8).
    """

    def __init__(self, group_size: int = 16):
        self.group_size = group_size

    # ------------------------------------------------------------------
    def quantize(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          q          : dequantized fp tensor (same shape)
          l1_scale   : scalar fp32 tensor
          l2_scales  : per-group fp8-simulated scales, shape [num_groups]
        """
        orig_shape = tensor.shape
        w = tensor.reshape(-1)
        n = w.numel()
        gs = self.group_size

        # --- L1 scale (per-tensor) ---
        abs_max = w.abs().amax().clamp(min=1e-30)
        l1_scale = (abs_max / 6.0).to(torch.float32)   # scalar

        w_norm = w / l1_scale                           # → roughly in [-6, 6]

        # pad to multiple of group_size
        pad = (gs - n % gs) % gs
        if pad:
            w_norm = torch.cat([w_norm, w_norm.new_zeros(pad)])

        w_groups = w_norm.reshape(-1, gs)               # [G, gs]

        # --- L2 scale (per-group, stored as FP8 E4M3) ---
        grp_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        l2_raw = (grp_max / 6.0).float()               # [G, 1]
        l2_fp8 = _to_fp8_e4m3(l2_raw)                  # simulate FP8 storage

        # --- quantize groups ---
        w_scaled = w_groups / l2_fp8                    # [G, gs]
        w_q = _nearest_e2m1(w_scaled)                  # [G, gs]

        # --- dequantize ---
        w_dq = (w_q * l2_fp8 * l1_scale).reshape(-1)[:n].reshape(orig_shape)

        return w_dq, l1_scale, l2_fp8.squeeze(-1)

    # ------------------------------------------------------------------
    def dequantize(
        self,
        q: torch.Tensor,
        l1_scale: torch.Tensor,
        l2_scales: torch.Tensor,
    ) -> torch.Tensor:
        # q already contains the fully dequantized float values
        return q

    # ------------------------------------------------------------------
    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-quantize (training): simulate two-level scaling noise."""
        orig_shape = tensor.shape
        w = tensor.reshape(-1)
        n = w.numel()
        gs = self.group_size

        abs_max = w.abs().amax().clamp(min=1e-30)
        l1_scale = abs_max / 6.0

        w_norm = w / l1_scale

        pad = (gs - n % gs) % gs
        if pad:
            w_norm = torch.cat([w_norm, w_norm.new_zeros(pad)])

        w_groups = w_norm.reshape(-1, gs)

        grp_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
        l2_fp8 = _to_fp8_e4m3(grp_max / 6.0)

        w_q = _nearest_e2m1(w_groups / l2_fp8)
        w_dq = (w_q * l2_fp8 * l1_scale).reshape(-1)[:n].reshape(orig_shape)
        return w_dq
