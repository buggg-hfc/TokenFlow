"""
Base classes for custom quantization.

Two modes:
  - fake_quant=True  (training): quantize then dequantize in fp32/bf16,
                                 gradients flow through via STE.
  - fake_quant=False (inference): store weights in packed int4, decode on forward.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Straight-Through Estimator (STE)
# Passes gradient through the quantization step unchanged.
# ---------------------------------------------------------------------------
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_q):
        return x_q

    @staticmethod
    def backward(ctx, grad):
        return grad, None  # pass gradient straight through


def ste(x: torch.Tensor, x_q: torch.Tensor) -> torch.Tensor:
    """Replace x with x_q in forward, use grad of x in backward."""
    return STEFunction.apply(x, x_q)


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------
@dataclass
class QuantizationConfig:
    method: str = "mxfp4"           # mxfp4 | nvfp4 | hif4
    fake_quant: bool = True          # True=training, False=inference
    weight_quant: bool = True        # quantize weights
    act_quant: bool = False          # quantize activations (input to Linear)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    # method-specific overrides (passed to quantizer __init__)
    method_kwargs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract quantizer
# ---------------------------------------------------------------------------
class BaseQuantizer(ABC):
    """
    Subclass this to implement a new FP4/INT4 format.

    Contract:
      quantize(tensor)  -> (q_tensor, *scales)   # q_tensor is still float (fake) or int (real)
      dequantize(q, *scales) -> tensor            # back to bf16/fp32
      quantize_dequantize(tensor) -> tensor       # one-shot for fake_quant training
    """

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns (quantized_tensor, *scale_tensors)."""

    @abstractmethod
    def dequantize(self, q: torch.Tensor, *scales) -> torch.Tensor:
        """Reconstruct fp tensor from quantized representation."""

    def quantize_dequantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-quantize: apply quant noise but stay in float domain."""
        return self.dequantize(*self.quantize(tensor))


# ---------------------------------------------------------------------------
# Quantized Linear layer
# ---------------------------------------------------------------------------
class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that applies custom quantization.

    fake_quant=True  (training) : weights stay float, quantize_dequantize()
                                  is called each forward to simulate quant noise.
    fake_quant=False (inference): weights stored as packed int4 + scales,
                                  dequantized on each forward.
    """

    def __init__(
        self,
        linear: nn.Linear,
        quantizer: "BaseQuantizer",
        fake_quant: bool = True,
        act_quant: bool = False,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quantizer = quantizer
        self.fake_quant = fake_quant
        self.act_quant = act_quant
        self.bias = linear.bias  # keep bias as-is (fp)

        if fake_quant:
            # Keep full-precision weights as trainable parameter
            self.weight = nn.Parameter(linear.weight.data.clone())
        else:
            # Pre-quantize and store packed representation
            self.register_buffer("weight", None)  # placeholder
            self._store_quantized(linear.weight.data)

    def _store_quantized(self, weight: torch.Tensor):
        """Quantize weight once and store scales + packed codes."""
        q, *scales = self.quantizer.quantize(weight)
        self.register_buffer("weight_q", q)
        for i, s in enumerate(scales):
            self.register_buffer(f"scale_{i}", s)
        self._num_scales = len(scales)

    def _get_dequantized_weight(self) -> torch.Tensor:
        if self.fake_quant:
            # Fake quant: quantize_dequantize with STE so gradients flow
            w_q = self.quantizer.quantize_dequantize(self.weight)
            return ste(self.weight, w_q)
        else:
            scales = [getattr(self, f"scale_{i}") for i in range(self._num_scales)]
            return self.quantizer.dequantize(self.weight_q, *scales)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._get_dequantized_weight()
        if self.act_quant:
            x_q = self.quantizer.quantize_dequantize(x)
            x = ste(x, x_q)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"method={self.quantizer.__class__.__name__}, "
                f"fake_quant={self.fake_quant}")
