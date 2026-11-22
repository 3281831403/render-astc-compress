"""Regression tests for the 4x4 block codec.

Every test exercises the public stubs in ``astc.encode``.  All tests are
expected to fail with ``NotImplementedError`` in the initial repository.
"""

from astc.encode import (
    block_bounds,
    encode_image,
    endpoints,
    pack_block,
    quantize_endpoints,
    unpack_block,
    validate_stream,
    weights_for,
)


def _image(w, h):
    """Deterministic, non-random test image with channel variation."""
    return [
        [
            ((x * 37 + y * 11) % 256, (x * 17 + y * 53) % 256, (x * 97 + y * 29) % 256)
            for x in range(w)
        ]
        for y in range(h)
    ]


def _decode_pixels(e0, e1, weights):
    """Integer interpolation used by the stream format."""
    return [
        tuple((e0[c] * (255 - w) + e1[c] * w + 127) // 255 for c in range(3))
        for w in weights
    ]


def _decode_block(data):
    parsed = unpack_block(data)
    return _decode_pixels(parsed["e0"], parsed["e1"], parsed["weights"])


def _decode_image(data, w, h, bw, bh):
    """Decode a stream into the full padded block grid."""
    across = (w + bw - 1) // bw
    down = (h + bh - 1) // bh
    grid = [[None] * (across * bw) for _ in range(down * bh)]
    for by in range(down):
        for bx in range(across):
            offset = (by * across + bx) * 16
            block = _decode_block(data[offset : offset + 16])
            for yy in range(bh):
                for xx in range(bw):
                    grid[by * bh + yy][bx * bw + xx] = block[yy * bw + xx]
    return grid


def test_block_bounds_are_half_open():
    assert block_bounds(0, 0, 4, 4) == (0, 0, 4, 4)
    assert block_bounds(4, 8, 4, 4) == (4, 8, 8, 12)
    assert block_bounds(12, 0, 4, 4) == (12, 0, 16, 4)


def test_endpoints_are_true_channel_extrema():
    block = [
        (3, 20, 30),
        (250, 7, 199),
        (10, 100, 9),
        (20, 200, 40),
        (30, 20, 50),
        (40, 30, 60),
        (50, 40, 70),
        (60, 50, 80),
        (70, 60, 90),
        (80, 70, 100),
        (90, 80, 110),
        (100, 90, 120),
        (110, 100, 130),
        (120, 110, 140),
        (130, 120, 150),
        (140, 130, 160),
    ]
    assert endpoints(block) == ((3, 7, 9), (250, 200, 199))


def test_quantize_endpoints_uses_bit_replication():
    lo, hi = quantize_endpoints(((0, 0, 0), (255, 255, 255)), 5)
    assert lo == (0, 0, 0)
    assert hi == (255, 255, 255)

    # 8 is exactly the 5-bit code 00001 replicated; 247 is 11110 replicated.
    lo, hi = quantize_endpoints(((8, 8, 8), (247, 247, 247)), 5)
    assert lo == (8, 8, 8)
    assert hi == (247, 247, 247)

    # 4-bit replication is also exact at both ends.
    lo, hi = quantize_endpoints(((0, 0, 0), (255, 255, 255)), 4)
    assert lo == (0, 0, 0)
    assert hi == (255, 255, 255)


def test_weights_for_is_normalized_and_clamped():
    block = [
        (0, 0, 0),
        (255, 255, 255),
        (128, 128, 128),
        (255, 0, 0),
    ] + [(64, 64, 64)] * 12
    e = ((0, 0, 0), (255, 255, 255))
    weights = weights_for(block, e)

    assert len(weights) == 16
    assert all(isinstance(w, int) and 0 <= w <= 255 for w in weights)
    assert weights[0] == 0
    assert weights[1] == 255
    assert weights[2] == 128
    assert weights[3] == 85


def test_pack_block_respects_weight_bit_width():
    weights = [0, 85, 170, 255] * 4
    data = pack_block((1, 2, 3), (4, 5, 6), weights, 2)

    assert len(data) == 16
    assert data[6] == 1  # bits - 1
    parsed = unpack_block(data)
    assert parsed["bits"] == 2
    assert parsed["weights"] == weights


def test_unpack_block_preserves_endpoint_order():
    e0 = (10, 20, 30)
    e1 = (200, 210, 220)
    data = pack_block(e0, e1, [0] * 16, 3)
    parsed = unpack_block(data)

    assert parsed["e0"] == e0
    assert parsed["e1"] == e1
    first = _decode_pixels(parsed["e0"], parsed["e1"], parsed["weights"])[0]
    assert first == e0


def test_encode_image_pads_tail_block_edges():
    img = _image(5, 4)
    data = encode_image(img, 4, 4)
    grid = _decode_image(data, 5, 4, 4, 4)

    # x=4 is the last real column; x=5..7 must replicate it.
    for y in range(4):
        assert grid[y][4] == grid[y][5] == grid[y][6] == grid[y][7]


def test_encode_image_uses_ceiling_block_count():
    img = _image(5, 4)
    data = encode_image(img, 4, 4)
    assert len(data) == 32  # ceil(5/4) * ceil(4/4) = 2 blocks


def test_validate_stream_rejects_garbage_padding():
    img = _image(4, 4)
    data = encode_image(img, 4, 4)
    assert validate_stream(data, 4, 4, 4, 4) == []

    corrupted = bytearray(data)
    corrupted[15] = 0x5A
    assert validate_stream(bytes(corrupted), 4, 4, 4, 4) != []


def test_pack_block_bit_order_is_little_endian():
    weights = [0, 85, 170, 255] * 4
    data = pack_block((0, 0, 0), (255, 255, 255), weights, 2)

    assert data[:6] == bytes([0, 0, 0, 255, 255, 255])
    assert data[6] == 1
    # 2-bit codes 0,1,2,3 in consecutive low-to-high bit positions.
    assert data[7:11] == bytes([0xE4, 0xE4, 0xE4, 0xE4])
    assert data[11:15] == b"\x00\x00\x00\x00"
    assert data[15] == 0


def test_single_block_roundtrip_pixel_error_at_most_one():
    e0 = (16, 32, 48)
    e1 = (24, 40, 56)
    block = []
    for i in range(16):
        t = i * 17
        block.append(
            tuple((e0[c] * (255 - t) + e1[c] * t + 127) // 255 for c in range(3))
        )

    weights = weights_for(block, (e0, e1))
    data = pack_block(e0, e1, weights, 4)
    parsed = unpack_block(data)
    rebuilt = _decode_pixels(parsed["e0"], parsed["e1"], parsed["weights"])

    for got, want in zip(rebuilt, block):
        assert max(abs(a - b) for a, b in zip(got, want)) <= 1


def test_solid_block_roundtrip_is_exact():
    color = (37, 99, 201)
    block = [color] * 16
    e = endpoints(block)
    assert e[0] == color
    assert e[1] == color

    weights = weights_for(block, e)
    data = pack_block(e[0], e[1], weights, 4)
    parsed = unpack_block(data)
    rebuilt = _decode_pixels(parsed["e0"], parsed["e1"], parsed["weights"])

    assert parsed["weights"] == [0] * 16
    assert rebuilt == block


def test_encoded_length_is_sixteen_times_block_count():
    img = _image(9, 7)
    data = encode_image(img, 4, 4)

    across = (9 + 4 - 1) // 4
    down = (7 + 4 - 1) // 4
    assert len(data) == 16 * across * down


def test_tail_padding_block_decodes_without_out_of_bounds():
    img = _image(6, 3)
    data = encode_image(img, 4, 4)
    assert len(data) == 32

    tail = unpack_block(data[16:32])
    assert tail["bits"] == 4
    assert len(tail["weights"]) == 16
    assert len(tail["e0"]) == 3
    assert len(tail["e1"]) == 3

    grid = _decode_image(data, 6, 3, 4, 4)
    assert len(grid) == 4
    assert all(len(row) == 8 for row in grid)
    assert all(0 <= c <= 255 for row in grid for px in row for c in px)


def test_two_blocks_encode_identically_to_independent_blocks():
    left = _image(4, 4)
    right = [
        [((r + 101) % 256, (g + 53) % 256, (b + 17) % 256) for (r, g, b) in row]
        for row in _image(4, 4)
    ]
    combined = [left[y] + right[y] for y in range(4)]

    assert encode_image(combined, 4, 4) == encode_image(left, 4, 4) + encode_image(
        right, 4, 4
    )
