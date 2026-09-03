"""W8A8_MXFP8 (E8M0 scale) layout helpers and shared constants for Ascend NPU.

This module MUST NOT import ``torch_npu`` at top level: it is imported lazily
from shared modules (e.g. ``model_loader/per_block_fp8_quant_weight.py``)
that are unconditionally imported on every platform. ``get_e8m0_dtype``
resolves the dtype lazily at call time (callers are NPU-only modules).

Reference: vllm-ascend ``quantization/methods/w8a8_mxfp8.py``
``process_weights_after_loading``:
  2D dense: [N, kp]    -> [kp_pad // 2, N, 2]
  3D MoE:   [E, N, kp] -> [E, kp_pad // 2, N, 2]
"""

import torch
import torch.nn.functional as F

__all__ = ["MXFP8_GROUP_SIZE", "get_e8m0_dtype", "swizzle_scale_to_npu_layout"]

# MXFP8 block size: one E8M0 scale per 32 elements along K.
MXFP8_GROUP_SIZE = 32


def get_e8m0_dtype():
    """Logical dtype of E8M0 scales (prefer torch_npu, fall back to torch)."""
    try:
        import torch_npu  # lazy: keep this module importable on non-NPU hosts

        dtype = getattr(torch_npu, "float8_e8m0fnu", None)
        if dtype is not None:
            return dtype
    except ImportError:
        pass
    return getattr(torch, "float8_e8m0fnu", None)


def swizzle_scale_to_npu_layout(scale: torch.Tensor) -> torch.Tensor:
    """Swizzle E8M0 scales into the pair-split layout required by npu_quant_matmul.

    Executed in ``PerBlockFp8Weight._postprocess`` (i.e. AFTER the TP split),
    because the pair-split layout spans the whole K-group dimension and an
    earlier swizzle would be broken by the split.

    Args:
        scale: E8M0 scales stored as uint8.
            2D dense: ``[N, kp]``; 3D MoE: ``[E, N, kp]`` where ``kp = K // 32``.

    Returns:
        2D: ``[kp_pad // 2, N, 2]``; 3D: ``[E, kp_pad // 2, N, 2]``.
        ``kp`` is zero-padded to even when odd (the padded pair column belongs
        to no weight group and never participates in numerics).
        Only memory layout is permuted; E8M0 values are unchanged.
    """
    if scale.dim() == 2:
        n, kp = scale.shape
        if kp % 2 != 0:
            scale = F.pad(scale, (0, 1))
            kp += 1
        # [N, kp] -> [N, kp//2, 2] -> [kp//2, N, 2]
        return scale.reshape(n, kp // 2, 2).transpose(0, 1).contiguous()

    if scale.dim() == 3:
        e, n, kp = scale.shape
        if kp % 2 != 0:
            scale = F.pad(scale, (0, 1))
            kp += 1
        # [E, N, kp] -> [E, N, kp//2, 2] -> [E, kp//2, N, 2]
        return scale.reshape(e, n, kp // 2, 2).transpose(1, 2).contiguous()

    raise ValueError(
        f"swizzle_scale_to_npu_layout expects 2D/3D scale, got shape {tuple(scale.shape)}"
    )
