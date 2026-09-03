"""Ascend MoE strategies"""

from .pytorch_fallback import AscendBf16FallbackStrategy
from .w8a8_mxfp8_strategy import AscendW8A8MXFP8MoeStrategy

__all__ = [
    "AscendBf16FallbackStrategy",
    "AscendW8A8MXFP8MoeStrategy",
]
