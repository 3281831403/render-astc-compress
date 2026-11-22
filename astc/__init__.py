"""ASTC-style 4x4 block compression exercise package."""

from .encode import (
    BLOCK_BYTES,
    BLOCK_PIXELS,
    DEFAULT_WEIGHT_BITS,
    MAX_WEIGHT_BITS,
    block_bounds,
    encode_image,
    endpoints,
    pack_block,
    quantize_endpoints,
    unpack_block,
    validate_stream,
    weights_for,
)

__all__ = [
    "BLOCK_BYTES",
    "BLOCK_PIXELS",
    "DEFAULT_WEIGHT_BITS",
    "MAX_WEIGHT_BITS",
    "block_bounds",
    "encode_image",
    "endpoints",
    "pack_block",
    "quantize_endpoints",
    "unpack_block",
    "validate_stream",
    "weights_for",
]
