"""Small, deterministic 4x4 block codec used by the render-astc exercise.

The public API works with ordinary 8-bit RGB values.  A block is exactly 16
pixels in row-major order, each pixel being ``(r, g, b)`` with channels in
``0..255``.  An encoded block is exactly 16 bytes.

Block byte layout (all offsets are byte offsets)::

    0..2    endpoint e0, raw r/g/b bytes
    3..5    endpoint e1, raw r/g/b bytes
    6       header: low 3 bits = weight_bits - 1, high 5 bits zero
    7..14   sixteen weight codes, LSB-first; code 0 is bit 0 of byte 7
    15      reserved padding byte, always zero

For ``weight_bits`` in ``1..4`` the 16 codes occupy ``16 * weight_bits``
bits.  Bits after that, and the whole of byte 15, are zero.  Codes are packed
little-endian at bit level: bit 0 is the least significant bit of byte 7,
bit 7 is the most significant bit of byte 7, bit 8 is bit 0 of byte 8, etc.

Weights supplied to ``pack_block`` always use the 0..255 scale, where 0 means
e0 and 255 means e1.  Packing rescales a weight ``w`` to a code::

    q = (w * maxw + 127) // 255            # maxw = (1 << weight_bits) - 1

and unpacking expands it back to the 0..255 scale::

    w = (q * 255 + maxw // 2) // maxw

Pixel reconstruction is the integer interpolation::

    c = (e0 * (255 - w) + e1 * w + 127) // 255

Round-trip contract: for a 4x4 block whose pixels lie on the e0->e1 line
with a per-channel span of at most 8, encoding with four weight bits must
reconstruct every channel within 1 (8-bit unit); a solid colour block must
round-trip with zero error.
"""

BLOCK_BYTES = 16
BLOCK_PIXELS = 16
DEFAULT_WEIGHT_BITS = 4
MAX_WEIGHT_BITS = 4


def block_bounds(x, y, bw, bh):
    """Return the half-open pixel bounds of one block.

    ``x`` and ``y`` are the pixel coordinates of the block's top-left corner
    (not block indices).  ``bw`` and ``bh`` are the block width and height.

    The result is ``(x0, y0, x1, y1)`` where ``x0 == x``, ``y0 == y``,
    ``x1 == x + bw`` and ``y1 == y + bh``.  The right and bottom edges are
    exclusive, so a 4-pixel-wide block at x=4 covers columns 4, 5, 6 and 7
    and returns ``(4, y, 8, y + bh)``.
    """
    return (x, y, x + bw, y + bh)


def endpoints(block):
    """Return the true per-channel colour endpoints of a 16-pixel block.

    ``block`` is a sequence of 16 ``(r, g, b)`` pixels in row-major order.
    The result is ``(e0, e1)`` where ``e0`` holds the minimum red, green and
    blue values found anywhere in the block, and ``e1`` holds the maximum
    values.  Each endpoint is an ``(r, g, b)`` tuple of 8-bit integers.

    The extrema are computed independently per channel; they need not come
    from the same pixel.
    """
    e0 = tuple(min(pixel[c] for pixel in block) for c in range(3))
    e1 = tuple(max(pixel[c] for pixel in block) for c in range(3))
    return e0, e1


def _replicate_bits(q, bits):
    """Expand an unsigned ``bits``-bit code to 8 bits by bit replication.

    The code's bits are repeated from most significant to least significant
    until eight bits are filled, so code 0 always maps to 0 and the largest
    code always maps to exactly 255.
    """
    result = 0
    for i in range(8):
        source_bit = bits - 1 - (i % bits)
        result |= ((q >> source_bit) & 1) << (7 - i)
    return result


def quantize_endpoints(e, bits):
    """Quantize a pair of 8-bit endpoints and expand them back to 8 bits.

    ``e`` is ``(e0, e1)``, each an ``(r, g, b)`` tuple with channels in
    ``0..255``.  ``bits`` is the number of stored bits per channel and must be
    in ``1..8``.

    For a channel value ``v``, the stored code is
    ``q = (v * maxq + 127) // 255`` with ``maxq = (1 << bits) - 1``.  The
    decoder-side value is produced by *bit replication*: the ``bits`` bits of
    ``q`` are repeated from most significant to least significant until eight
    bits exist, and the first eight bits form the result.  Consequently the
    smallest code maps to 0 and the largest code maps to exactly 255.

    The return value is ``(q0, q1)``, a pair of expanded 8-bit endpoints.
    ``bits == 8`` is an identity operation.
    """
    maxq = (1 << bits) - 1
    expanded = []
    for endpoint in e:
        out = []
        for v in endpoint:
            q = (v * maxq + 127) // 255
            out.append(_replicate_bits(q, bits))
        expanded.append(tuple(out))
    return tuple(expanded)


def weights_for(block, e):
    """Project every block pixel onto the e0->e1 line and return weights.

    ``block`` is 16 row-major ``(r, g, b)`` pixels.  ``e`` is the endpoint
    pair ``(e0, e1)``, each already at 8-bit precision.

    Let ``d = e1 - e0`` and ``den = dot(d, d)``.  For pixel ``p`` the scalar
    projection is ``t = dot(p - e0, d) / den``.  If ``den == 0`` (a solid
    block) every weight is 0.  Otherwise ``t`` is clamped to ``0..1`` and the
    returned weight is the integer ``round(t * 255)``, i.e.
    ``(t * 255)`` rounded half-up.

    The result is a list of exactly 16 integers, each in ``0..255``.  Weight
    0 selects ``e0`` and weight 255 selects ``e1``.
    """
    e0, e1 = e
    d = tuple(e1[c] - e0[c] for c in range(3))
    den = sum(v * v for v in d)
    if den == 0:
        return [0] * BLOCK_PIXELS

    weights = []
    for pixel in block:
        num = sum((pixel[c] - e0[c]) * d[c] for c in range(3))
        if num <= 0:
            weights.append(0)
        elif num >= den:
            weights.append(255)
        else:
            # Half-up rounding of 255 * num / den, done in integer arithmetic.
            weights.append((2 * 255 * num + den) // (2 * den))
    return weights


def pack_block(e0, e1, weights, bits):
    """Pack one 4x4 block into exactly 16 bytes.

    ``e0`` and ``e1`` are ``(r, g, b)`` endpoints with channels in ``0..255``
    and are stored losslessly as the first six bytes.  ``weights`` is a
    sequence of 16 integers on the 0..255 scale.  ``bits`` is the weight code
    width and must be in ``1..4``.

    Each weight is rescaled with
    ``q = (w * maxw + 127) // 255`` and the sixteen codes are written
    LSB-first starting at bit 0 of byte 7.  Byte 6 stores ``bits - 1`` in its
    low three bits; its high five bits are zero.  Byte 15 and every unused
    high bit after the last code are zero.

    The return value is a ``bytes`` object of length 16.
    """
    if not 1 <= bits <= MAX_WEIGHT_BITS:
        raise ValueError("bits must be in 1..4")
    if len(weights) != BLOCK_PIXELS:
        raise ValueError("exactly 16 weights are required")

    data = bytearray(BLOCK_BYTES)
    data[0:3] = bytes(e0)
    data[3:6] = bytes(e1)
    data[6] = bits - 1

    maxw = (1 << bits) - 1
    bitpos = 0  # Bit offset relative to bit 0 of byte 7.
    for w in weights:
        q = (w * maxw + 127) // 255
        for b in range(bits):
            if (q >> b) & 1:
                data[7 + (bitpos >> 3)] |= 1 << (bitpos & 7)
            bitpos += 1
    return bytes(data)


def unpack_block(data):
    """Decode one 16-byte block produced by :func:`pack_block`.

    The header supplies the weight width.  The result is a dictionary with
    exactly these keys:

    ``"e0"``
        The first endpoint as an ``(r, g, b)`` tuple of 8-bit integers.
    ``"e1"``
        The second endpoint as an ``(r, g, b)`` tuple of 8-bit integers.
    ``"weights"``
        A list of 16 expanded integers in ``0..255``.
    ``"bits"``
        The weight code width from the header (``1..4``).

    Endpoint order is the same as the order used by ``pack_block``: weight 0
    selects ``e0`` and the maximum weight selects ``e1``.
    """
    e0 = tuple(data[0:3])
    e1 = tuple(data[3:6])
    bits = (data[6] & 0x07) + 1
    maxw = (1 << bits) - 1

    weights = []
    bitpos = 0  # Bit offset relative to bit 0 of byte 7.
    for _ in range(BLOCK_PIXELS):
        q = 0
        for b in range(bits):
            q |= ((data[7 + (bitpos >> 3)] >> (bitpos & 7)) & 1) << b
            bitpos += 1
        weights.append((q * 255 + maxw // 2) // maxw)

    return {"e0": e0, "e1": e1, "weights": weights, "bits": bits}


def encode_image(img, bw, bh):
    """Encode a rectangular RGB image as concatenated 16-byte blocks.

    ``img`` is a non-empty sequence of equal-length rows, each row a sequence
    of ``(r, g, b)`` pixels.  Only 4x4 blocks are supported, so ``bw`` and
    ``bh`` are both 4.

    The block grid uses ceiling division: the number of blocks across is
    ``(width + bw - 1) // bw`` and down is ``(height + bh - 1) // bh``.
    Pixels outside the image are filled by replicating the nearest in-bounds
    edge pixel (clamp x to ``width - 1`` and y to ``height - 1``); no
    out-of-range read is allowed.  Each block is compressed with
    ``DEFAULT_WEIGHT_BITS`` (4) weight bits.

    The return value is ``16 * nblocks`` bytes in row-major block order, with
    each block in the layout documented for :func:`pack_block`.
    """
    height = len(img)
    width = len(img[0])
    across = (width + bw - 1) // bw
    down = (height + bh - 1) // bh

    stream = bytearray()
    for by in range(down):
        for bx in range(across):
            block = []
            for yy in range(bh):
                for xx in range(bw):
                    # Clamp to the last in-bounds edge pixel (edge replication).
                    x = min(bx * bw + xx, width - 1)
                    y = min(by * bh + yy, height - 1)
                    block.append(img[y][x])
            e0, e1 = endpoints(block)
            weights = weights_for(block, (e0, e1))
            stream.extend(
                pack_block(e0, e1, weights, DEFAULT_WEIGHT_BITS)
            )
    return bytes(stream)


def validate_stream(data, w, h, bw, bh):
    """Validate an encoded stream for image dimensions ``w`` by ``h``.

    Return a list of human-readable error strings; an empty list means the
    stream is valid.  The checks are:

    * ``w``, ``h``, ``bw`` and ``bh`` are positive;
    * the stream length is a non-zero multiple of 16;
    * the number of blocks is exactly
      ``ceil(w / bw) * ceil(h / bh)`` (no missing or extra blocks);
    * every header stores a weight width in ``1..4`` with reserved header
      bits zero;
    * every reserved padding byte is zero, and all unused bits after the last
      weight code are zero;
    * each block decodes to 16 weights and two 8-bit endpoints; and
    * for partial right/bottom blocks, every out-of-image pixel reconstructed
      from the stream equals the reconstructed in-bounds edge pixel it was
      padded from (edge replication).

    The function reports errors instead of raising for malformed streams.
    """
    errors = []
    if min(w, h, bw, bh) <= 0:
        errors.append("w, h, bw and bh must all be positive")
        return errors

    if len(data) == 0 or len(data) % BLOCK_BYTES != 0:
        errors.append("stream length must be a non-zero multiple of 16 bytes")

    across = (w + bw - 1) // bw
    down = (h + bh - 1) // bh
    expected_blocks = across * down
    nblocks = len(data) // BLOCK_BYTES
    if nblocks != expected_blocks:
        errors.append(
            f"expected {expected_blocks} blocks for {w}x{h}, got {nblocks}"
        )

    for i in range(min(nblocks, expected_blocks)):
        blk = data[i * BLOCK_BYTES:(i + 1) * BLOCK_BYTES]
        header = blk[6]

        if header & 0xF8:
            errors.append(f"block {i}: reserved header bits are non-zero")

        bits = (header & 0x07) + 1
        if not 1 <= bits <= MAX_WEIGHT_BITS:
            errors.append(f"block {i}: weight width {bits} is outside 1..4")
            continue

        if blk[15] != 0:
            errors.append(f"block {i}: reserved padding byte 15 is non-zero")

        # All bits beyond the 16 packed codes must be zero.
        used_bits = BLOCK_PIXELS * bits
        for bit in range(used_bits, 64):
            if (blk[7 + (bit >> 3)] >> (bit & 7)) & 1:
                errors.append(f"block {i}: unused weight bits are non-zero")
                break

        # Decode the block exactly like unpack_block.
        maxw = (1 << bits) - 1
        weights = []
        bitpos = 0
        for _ in range(BLOCK_PIXELS):
            q = 0
            for b in range(bits):
                q |= ((blk[7 + (bitpos >> 3)] >> (bitpos & 7)) & 1) << b
                bitpos += 1
            weights.append((q * 255 + maxw // 2) // maxw)

        e0 = tuple(blk[0:3])
        e1 = tuple(blk[3:6])
        pixels = [
            tuple(
                (e0[c] * (255 - wt) + e1[c] * wt + 127) // 255
                for c in range(3)
            )
            for wt in weights
        ]

        # Partial blocks: out-of-image pixels must replicate the edge pixel.
        by = i // across
        bx = i % across
        for yy in range(bh):
            for xx in range(bw):
                gx = bx * bw + xx
                gy = by * bh + yy
                if gx >= w or gy >= h:
                    src_x = min(gx, w - 1) - bx * bw
                    src_y = min(gy, h - 1) - by * bh
                    if pixels[yy * bw + xx] != pixels[src_y * bw + src_x]:
                        errors.append(
                            f"block {i}: padding pixel ({gx}, {gy}) does not "
                            "replicate its in-bounds edge pixel"
                        )

    return errors
