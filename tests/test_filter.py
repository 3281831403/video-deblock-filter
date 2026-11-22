"""Red tests for H.264-like luma deblocking.

The tests are the evaluation contract: fix ``deblock/filter.py`` rather than
editing this file.  Every test currently fails with ``NotImplementedError``
from one of the eight stubs.  Frames are 8-bit grayscale ``list[list[int]]``
images addressed as ``frame[y][x]``; numeric comparisons use an absolute
tolerance of ``0.5`` pixel unless a tighter tolerance is stated.
"""

import pytest

import deblock.filter as df


def _copy(frame):
    return [list(row) for row in frame]


def _constant(h, w, value):
    return [[value] * w for _ in range(h)]


def _pattern(h, w, a=17, b=31):
    return [[(a * x + b * y) % 256 for x in range(w)] for y in range(h)]


def test_boundary_strength_uses_qp_scaled_thresholds_not_fixed_10():
    # The same 3-level contrast is bS=2 at low QP (alpha=4, beta=2),
    # but only bS=1 at high QP (alpha=80, beta=11).  A hard-coded 10 would
    # classify both cases identically.
    p = (100, 100, 100, 100)
    assert df.boundary_strength(p, (101, 101, 101, 101), 16, False) == 0
    assert df.boundary_strength(p, (103, 103, 103, 103), 16, False) == 2
    assert df.boundary_strength(p, (103, 103, 103, 103), 40, False) == 1

    # Four threshold levels 0..3 at QP 26: alpha=15, beta=4.
    assert df.boundary_strength(p, (103, 103, 103, 103), 26, False) == 1
    assert df.boundary_strength(p, (108, 108, 108, 108), 26, False) == 2
    assert df.boundary_strength(p, (120, 120, 120, 120), 26, False) == 3

    # Intra is the fourth / strongest level and is always 4.
    assert df.boundary_strength(p, (100, 100, 100, 100), 26, True) == 4


def test_bs4_edge_uses_strong_filter_not_weak_branch():
    p = (100, 110, 120, 130)
    q = (150, 160, 170, 180)

    strong_p, strong_q = df.filter_luma(p, q, 4, 10)
    weak_p, weak_q = df.filter_luma(p, q, 3, 10)

    # A strong edge must update p1 and q1, not just p0/q0.
    assert strong_p[2] != p[2]
    assert strong_q[1] != q[1]

    # A weak edge changes only the two samples touching the boundary.
    assert weak_p[:3] == p[:3]
    assert weak_q[2:] == q[2:]
    assert (weak_p[3], weak_q[0]) != (p[3], q[0])


def test_thresholds_adds_offset_before_table_lookup():
    # Exact H.264 Table 8-16 rows, not interpolated values.
    assert df.thresholds(16, 0) == (4, 2, 0)
    assert df.thresholds(24, 0) == (12, 4, 1)
    assert df.thresholds(26, 0) == (15, 4, 1)

    # offset is added to QP before the table lookup.
    assert df.thresholds(24, 2) == (15, 4, 1)
    assert df.thresholds(20, 6) == df.thresholds(26, 0)
    assert df.thresholds(26, -2) == df.thresholds(24, 0)

    # The index is clamped to [0, 51].
    assert df.thresholds(20, -20) == (0, 0, 0)


def test_strong_filter_clamps_to_neighbourhood_no_overshoot():
    # p3 is a far-away outlier.  The unclamped p1 average would be 32, but
    # the local p2/p1/p0/q0/q1 neighbourhood is all zero.
    p = (255, 0, 0, 0)
    q = (0, 0, 0, 0)

    new_p, new_q = df.filter_luma(p, q, 4, 10)

    assert new_p[2] == 0
    local_low = min(p[1], p[2], p[3], q[0], q[1])
    local_high = max(p[1], p[2], p[3], q[0], q[1])
    assert local_low <= new_p[3] <= local_high
    assert local_low <= new_q[0] <= local_high


def test_filter_edge_reads_snapshot_before_writing_q_side():
    frame = [[100, 100, 100, 100, 140, 140, 140, 140] for _ in range(8)]
    before = _copy(frame)

    out = df.filter_edge(frame, 4, 0, True, 30)

    # Pure operation: no q-side write may leak back into the input.
    assert frame == before

    # QP 30 gives tC0=1 (beta is 6).  A correct weak filter moves p0 to 101
    # and q0 to 139; reading p after writing q would produce different values.
    for row in range(4):
        assert out[row][3] == 101
        assert out[row][4] == 139
        assert out[row][2] == 100
        assert out[row][5] == 140
    for row in range(4, 8):
        assert out[row] == frame[row]

    # The vertical p tuple was read from the snapshot, and the q tuple from
    # the snapshot as well; neither side was contaminated by the other.
    assert out[0][3] == 101
    assert out[1][3] == 101
    assert out[0][4] == 139
    assert out[1][4] == 139


def test_filter_frame_does_not_double_filter_crossing_pixels(monkeypatch):
    # The 4-pixel block boundaries at x=4 and y=4 cross at a 2x2 corner.
    # Asymmetric values make vertical-only and horizontal-only results differ.
    frame = (
        [[20, 40, 60, 200, 100, 220, 240, 255] for _ in range(4)]
        + [[200, 220, 240, 100, 0, 40, 60, 80] for _ in range(4)]
    )
    qp_map = _constant(8, 8, 30)

    vertical_only = frame
    for y in (0, 4):
        vertical_only = df.filter_edge(vertical_only, 4, y, True, 30)

    horizontal_only = frame
    for x in (0, 4):
        horizontal_only = df.filter_edge(horizontal_only, x, 4, False, 30)

    calls = []
    real_edge = df.filter_edge

    def tracking_edge(frame_arg, x, y, vertical, qp):
        calls.append((x, y, vertical))
        return real_edge(frame_arg, x, y, vertical, qp)

    monkeypatch.setattr(df, "filter_edge", tracking_edge)
    full = df.filter_frame(frame, qp_map, 4)

    # Each of the four unique 4-sample segments is processed exactly once.
    assert len(calls) == 4
    assert set(calls) == {
        (4, 0, True),
        (4, 4, True),
        (0, 4, False),
        (4, 4, False),
    }

    # (x=4, y=3): vertical is nearest, so the vertical result is used.
    assert full[3][4] == vertical_only[3][4]
    # (x=3, y=4): horizontal is nearest, so the horizontal result is used.
    assert full[4][3] == horizontal_only[4][3]
    # A double-filtered corner would equal neither single-orientation result.
    assert vertical_only[3][4] != horizontal_only[3][4]


def test_edge_map_marks_only_block_aligned_boundaries():
    # A very strong image edge at x=2 is not block-aligned and must not appear.
    frame = [[0, 0, 255, 255, 255, 255, 255, 255] for _ in range(8)]

    expected = [[0, 0, 0, 0, 1, 0, 0, 0] for _ in range(4)]
    expected.append([1, 1, 1, 1, 1, 1, 1, 1])
    expected.extend([[0, 0, 0, 0, 1, 0, 0, 0] for _ in range(3)])

    assert df.edge_map(frame, 4) == expected


def test_validate_frame_rejects_block_not_dividing_dimensions():
    frame = _pattern(8, 8)
    qp_map = _constant(8, 8, 30)

    assert df.validate_frame(frame, qp_map, 4) == []

    errors = df.validate_frame(frame, qp_map, 3)
    assert errors
    assert any("divide" in message.lower() for message in errors)

    wide_frame = _pattern(8, 10)
    wide_qp = _constant(8, 10, 30)
    width_errors = df.validate_frame(wide_frame, wide_qp, 4)
    assert width_errors
    assert any("divide" in message.lower() for message in width_errors)


def test_deblock_batch_uses_each_frames_own_qp_map():
    frame = (
        [[20, 40, 60, 80, 200, 220, 240, 255] for _ in range(4)]
        + [[200, 220, 240, 255, 20, 40, 60, 80] for _ in range(4)]
    )
    low_qp = _constant(8, 8, 16)   # tC0 == 0 here: a no-op.
    high_qp = _constant(8, 8, 45)  # tC0 == 3 here: visibly filtered.

    result = df.deblock_batch(
        [frame, frame],
        {"block": 4, "qp_maps": [low_qp, high_qp]},
    )

    assert result["frames"][0] == frame
    assert result["frames"][1] != frame
    assert result["frames"][0] != result["frames"][1]


def test_filter_luma_uses_tc0_not_beta():
    # Directly passing tc=1 must limit the weak-filter movement to one pixel.
    p = (100, 100, 100, 100)
    q = (200, 200, 200, 200)
    new_p, new_q = df.filter_luma(p, q, 3, 1)
    assert abs(new_p[3] - 100) <= 1
    assert abs(new_q[0] - 200) <= 1

    # At QP 30 beta=6 but tC0=1.  The edge helper must pass tC0.
    assert df.thresholds(30, 0) == (25, 6, 1)
    frame = [[100, 100, 100, 100, 140, 140, 140, 140] for _ in range(8)]
    out = df.filter_edge(frame, 4, 0, True, 30)
    assert out[0][3] == 101
    assert out[0][4] == 139


def test_flat_frame_unchanged():
    frame = _constant(8, 8, 128)
    qp_map = _constant(8, 8, 30)

    out = df.filter_frame(frame, qp_map, 4)

    assert len(out) == len(frame)
    for row, out_row in zip(frame, out):
        assert len(out_row) == len(row)
        for before, after in zip(row, out_row):
            assert abs(after - before) <= 1e-9


def test_single_strong_boundary_contrast_decreases():
    frame = [[0, 0, 0, 0, 255, 255, 255, 255] for _ in range(8)]
    before = abs(frame[0][3] - frame[0][4])

    out = df.filter_edge(frame, 4, 0, True, 51)

    after = abs(out[0][3] - out[0][4])
    assert after < before
    assert after >= 0


def test_frame_changes_limited_to_boundary_plus_minus_three():
    frame = _pattern(12, 12, a=13, b=29)
    qp_map = _constant(12, 12, 45)

    out = df.filter_frame(frame, qp_map, 4)

    changed = 0
    for y in range(12):
        for x in range(12):
            if out[y][x] == frame[y][x]:
                continue
            changed += 1
            near_vertical = any(abs(x - bx) <= 3 for bx in (4, 8))
            near_horizontal = any(abs(y - by) <= 3 for by in (4, 8))
            assert near_vertical or near_horizontal, (x, y)

    assert changed > 0


def test_vertical_and_horizontal_edge_order_commutes():
    frame = _pattern(16, 16)

    first_vertical = df.filter_edge(frame, 4, 0, True, 40)
    vertical_then_horizontal = df.filter_edge(
        first_vertical, 8, 12, False, 40
    )

    first_horizontal = df.filter_edge(frame, 8, 12, False, 40)
    horizontal_then_vertical = df.filter_edge(
        first_horizontal, 4, 0, True, 40
    )

    assert vertical_then_horizontal == horizontal_then_vertical


def test_batch_matches_independent_frame_filtering():
    frame_a = _pattern(8, 8, a=17, b=31)
    frame_b = (
        [[220, 220, 220, 220, 20, 20, 20, 20] for _ in range(4)]
        + [[20, 20, 20, 20, 220, 220, 220, 220] for _ in range(4)]
    )
    qp_a = _constant(8, 8, 26)
    qp_b = _constant(8, 8, 44)

    batch = df.deblock_batch(
        [frame_a, frame_b],
        {"block": 4, "qp_maps": [qp_a, qp_b]},
    )
    expected_a = df.filter_frame(frame_a, qp_a, 4)
    expected_b = df.filter_frame(frame_b, qp_b, 4)

    assert batch["frames"] == [expected_a, expected_b]
    assert batch["count"] == 2
    assert len(batch["edge_maps"]) == 2
    assert batch["edge_maps"][0] == df.edge_map(frame_a, 4)
