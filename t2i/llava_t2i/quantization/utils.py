"""
Quantization utility helpers.

Public API
----------
get_quantizer(config)           → BaseQuantizer instance
apply_quantization(model, cfg)  → model with QuantizedLinear layers
remove_quantization(model)      → model with plain nn.Linear layers restored
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseQuantizer, QuantizedLinear, QuantizationConfig
from .mxfp4 import MXFP4Quantizer
from .nvfp4 import NVFP4Quantizer
from .hif4 import HiF4Quantizer


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_quantizer(config: QuantizationConfig) -> BaseQuantizer:
    """Return the appropriate quantizer instance for the given config."""
    method = config.method.lower()
    kwargs = config.method_kwargs or {}

    if method == "mxfp4":
        return MXFP4Quantizer(**kwargs)
    elif method == "nvfp4":
        return NVFP4Quantizer(**kwargs)
    elif method == "hif4":
        return HiF4Quantizer(**kwargs)
    else:
        raise ValueError(
            f"Unknown quantization method '{method}'. "
            "Supported: mxfp4, nvfp4, hif4"
        )


# ---------------------------------------------------------------------------
# Apply quantization to a model
# ---------------------------------------------------------------------------

def _module_name_matches(name: str, target_modules) -> bool:
    """True if the module name ends with any of the target suffixes."""
    return any(name == t or name.endswith(f".{t}") for t in target_modules)


def apply_quantization(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Replace target nn.Linear layers with QuantizedLinear in-place.

    Parameters
    ----------
    model  : the model to modify (mutated in-place and returned)
    config : QuantizationConfig specifying method, fake_quant, target_modules, …

    Returns
    -------
    The same model object with Linear layers replaced.

    Example
    -------
    >>> from llava_t2i.quantization import apply_quantization, QuantizationConfig
    >>> cfg = QuantizationConfig(method="mxfp4", fake_quant=True)
    >>> model = apply_quantization(model, cfg)
    """
    quantizer = get_quantizer(config)
    target = set(config.target_modules)

    replaced = 0
    # Collect replacements first to avoid mutating dict during iteration
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module's leaf name matches any target
            leaf = name.split(".")[-1]
            if not target or leaf in target or name in target:
                replacements[name] = QuantizedLinear(
                    linear=module,
                    quantizer=quantizer,
                    fake_quant=config.fake_quant,
                    act_quant=config.act_quant,
                )
                replaced += 1

    # Apply replacements by traversing the parent chain
    for full_name, new_module in replacements.items():
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    print(f"[Quantization] Replaced {replaced} Linear layers with "
          f"QuantizedLinear (method={config.method}, "
          f"fake_quant={config.fake_quant})")
    return model


# ---------------------------------------------------------------------------
# Remove quantization (restore plain nn.Linear)
# ---------------------------------------------------------------------------

def remove_quantization(model: nn.Module) -> nn.Module:
    """
    Replace QuantizedLinear layers back to plain nn.Linear in-place.

    Useful for exporting a fine-tuned model in full precision.

    Returns the same model object.
    """
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            # Reconstruct a plain Linear with the current (possibly trained) weight
            bias = module.bias
            if module.fake_quant:
                weight_data = module.weight.data
            else:
                # Dequantize stored weights
                weight_data = module._get_dequantized_weight().data

            linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=(bias is not None),
                device=weight_data.device,
                dtype=weight_data.dtype,
            )
            linear.weight = nn.Parameter(weight_data)
            if bias is not None:
                linear.bias = nn.Parameter(bias.data.clone())

            replacements[name] = linear

    for full_name, new_module in replacements.items():
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    print(f"[Quantization] Restored {len(replacements)} QuantizedLinear → nn.Linear")
    return model
