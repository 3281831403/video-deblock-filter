"""H.264-like luma deblocking kernels.

Shared contract
---------------
* A frame is a rectangular, row-major ``list[list[int]]`` addressed as
  ``frame[y][x]``.  Every luma sample is an 8-bit integer in ``[0, 255]``.
* ``bS`` is the boundary strength and is always in ``0..4``.  ``4`` is the
  intra (strongest) edge; ``1..3`` are inter edges of decreasing importance;
  ``0`` means that the edge must not be filtered.
* ``qp`` is an H.264 luma QP in ``0..51``.  A QP map has the same shape as the
  frame and stores one QP per pixel; the QP at the first q-side sample of an
  edge segment owns that segment.
* A block edge is a geometric property.  ``block`` is positive, a multiple of
  four, and divides both frame dimensions, so internal boundaries are at
  positive multiples of ``block``.
* Pixel comparisons use an absolute tolerance of ``0.5`` pixel.  The kernels
  themselves return integers in ``[0, 255]``.
* ``p`` is the four-sample vector ``(p3, p2, p1, p0)`` and ``q`` is
  ``(q0, q1, q2, q3)``; ``p0`` and ``q0`` are the samples adjacent to the
  boundary.  The same ordering is used by :func:`boundary_strength`,
  :func:`filter_luma` and :func:`filter_edge`.
"""

from __future__ import annotations

__all__ = [
    "boundary_strength",
    "deblock_batch",
    "edge_map",
    "filter_edge",
    "filter_frame",
    "filter_luma",
    "thresholds",
    "validate_frame",
]

Frame = list[list[int]]
QPMap = list[list[int]]

# H.264 Table 8-16 luma alpha / beta threshold tables, indexed by the
# clamped table index (QP + offset, 0..51).  These are exact lookups: there
# is no interpolation between neighbouring QP rows.
_ALPHA_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20, 22, 25, 28,
    32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144,
    162, 182, 203, 226, 255, 255, 255,
)
_BETA_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6,
    7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
    14, 14, 15, 15, 16, 16, 17,
)

# H.264 Table 8-17 luma tC0 table, indexed by the clamped table index.
_TC0_TABLE = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
    5, 5, 6, 6, 6, 6,
)


def _clamp(value: int, low: int, high: int) -> int:
    if value < low:
        return low
    if value > high:
        return high
    return value


def boundary_strength(
    p: tuple[int, int, int, int] | list[int],
    q: tuple[int, int, int, int] | list[int],
    qp: int,
    intra: bool = False,
) -> int:
    """Return the H.264-like boundary strength ``bS`` in ``0..4``.

    ``p = (p3, p2, p1, p0)`` and ``q = (q0, q1, q2, q3)`` are the four luma
    samples on each side of one boundary.  The decision uses QP-dependent
    thresholds, not a fixed pixel-difference constant:

    * ``intra`` is true: the edge is an intra strong edge, so return ``4``.
    * Let ``d = max(abs(pi - qi) for i in 0..3)`` and let
      ``alpha, beta, _ = thresholds(qp, 0)``.
    * ``d <= 1`` returns ``0``; otherwise ``d < beta`` returns ``1``,
      ``d < alpha`` returns ``2``, and ``d >= alpha`` returns ``3``.

    Thus the same contrast can have a different strength at different QPs,
    and a low-QP flat area must not be promoted to a filterable edge.  ``bS``
    is always in ``0..4``; results are integers.
    """
    if intra:
        return 4

    alpha, beta, _ = thresholds(qp, 0)
    d = max(abs(int(p[i]) - int(q[i])) for i in range(4))

    if d <= 1:
        return 0
    if d < beta:
        return 1
    if d < alpha:
        return 2
    return 3


def thresholds(qp: int, offset: int = 0) -> tuple[int, int, int]:
    """Return ``(alpha, beta, tC0)`` for luma QP plus an offset.

    The values are an exact lookup in the H.264 Table 8-16 luma threshold
    tables at the single index ``clamp(qp + offset, 0, 51)``; there is no
    interpolation between neighbouring QP rows.  ``offset`` is applied before
    the lookup, so ``thresholds(24, 2) == thresholds(26, 0)``.  Negative
    offsets clamp at index zero and large offsets clamp at index 51.

    ``alpha``, ``beta`` and ``tC0`` are non-negative integers, and the three
    returned thresholds are non-decreasing with the table index.
    """
    index = _clamp(int(qp) + int(offset), 0, 51)
    return _ALPHA_TABLE[index], _BETA_TABLE[index], _TC0_TABLE[index]


def filter_luma(
    p: tuple[int, int, int, int] | list[int],
    q: tuple[int, int, int, int] | list[int],
    bS: int,
    tc: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Filter one four-sample luma edge and return ``(new_p, new_q)``.

    ``p = (p3, p2, p1, p0)`` and ``q = (q0, q1, q2, q3)``.  ``tc`` is the
    luma ``tC0`` threshold, never ``beta``.  The branches are:

    * ``bS == 0`` or ``tc <= 0``: return the inputs unchanged.
    * ``bS == 4``: strong branch.  It updates ``p1, p0, q0, q1`` using the
      H.264 strong-filter averages.  Every updated sample is clamped to
      the local neighbourhood ``[min(p2, p1, p0, q0, q1),
      max(p2, p1, p0, q0, q1)]`` and to ``[0, 255]``, so a far-away ``p3`` or
      ``q3`` cannot pull an output outside its neighbourhood.
    * ``1 <= bS <= 3``: weak branch.  It updates only ``p0`` and ``q0`` with
      the H.264 weak-filter delta clipped to ``[-tc, tc]``.

    All returned samples are 8-bit integers in ``[0, 255]``.  The result is
    compared with an absolute tolerance of ``0.5`` pixel.
    """
    p3, p2, p1, p0 = (int(v) for v in p)
    q0, q1, q2, q3 = (int(v) for v in q)

    # No edge, or a zero/negative tC0: the edge is not touched.
    if bS == 0 or tc <= 0:
        return (p3, p2, p1, p0), (q0, q1, q2, q3)

    if bS == 4:
        # H.264 8.7.5.3.3 strong-filter averages.  Each output is then
        # clamped to the local five-sample neighbourhood so a remote p3/q3
        # outlier cannot cause overshoot.
        local_low = min(p2, p1, p0, q0, q1)
        local_high = max(p2, p1, p0, q0, q1)

        def neighbourhood_clamp(value: int) -> int:
            return _clamp(value, local_low, local_high)

        new_p1 = neighbourhood_clamp(
            (p2 + p1 + p0 + q0 + 2) >> 2
        )
        new_p0 = neighbourhood_clamp(
            (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3
        )
        new_q0 = neighbourhood_clamp(
            (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3
        )
        new_q1 = neighbourhood_clamp(
            (p0 + q0 + q1 + q2 + 2) >> 2
        )
        return (
            (p3, p2, new_p1, new_p0),
            (new_q0, new_q1, q2, q3),
        )

    # 1 <= bS <= 3: weak branch.  Only the two boundary samples move, and
    # the movement is clipped to [-tc, tc] using tC0, never beta.  The raw
    # delta is the H.264 8.7.5.3.2 weak-filter delta; Python's >> on signed
    # integers is an arithmetic (sign-extending) shift, matching the spec.
    delta_raw = (((q0 - p0) << 2) + (p1 - q1) + 4) >> 3
    delta = _clamp(delta_raw, -tc, tc)
    new_p0 = _clamp(p0 + delta, 0, 255)
    new_q0 = _clamp(q0 - delta, 0, 255)
    return (p3, p2, p1, new_p0), (new_q0, q1, q2, q3)


def filter_edge(
    frame: Frame,
    x: int,
    y: int,
    vertical: bool,
    qp: int,
) -> list[list[int]]:
    """Return a copy of ``frame`` with one four-pixel edge segment filtered.

    ``(x, y)`` is the first ``q0`` sample of the segment.  For a vertical
    edge, ``q0`` is at ``(x, y)`` and ``p0`` at ``(x-1, y)``; the segment
    covers rows ``y..y+3`` and reads ``p3..p0`` from columns ``x-4..x-1`` and
    ``q0..q3`` from columns ``x..x+3``.  For a horizontal edge, ``q0`` is at
    ``(x, y)`` and ``p0`` at ``(x, y-1)``; the segment covers columns
    ``x..x+3`` and reads ``p3..p0`` from rows ``y-4..y-1`` and ``q0..q3``
    from rows ``y..y+3``.

    Each of the four positions uses :func:`boundary_strength` with
    ``intra=False`` and :func:`thresholds(qp, 0)``; the weak/strong result
    comes from :func:`filter_luma` with ``tC0`` as ``tc``.  All source samples
    are read from the input snapshot before any q-side sample is written, so
    the operation is pure: the input frame is never mutated and no q write can
    contaminate a later p read.

    The original frame is rectangular and the whole four-pixel segment must be
    in range; otherwise ``ValueError`` is raised.
    """
    height = len(frame)
    width = len(frame[0]) if height else 0

    if vertical:
        in_range = (
            0 <= y and y + 3 < height
            and 0 <= x - 4 and x + 3 < width
        )
    else:
        in_range = (
            0 <= x and x + 3 < width
            and 0 <= y - 4 and y + 3 < height
        )
    if not in_range:
        raise ValueError(
            "edge segment is out of range for the frame"
        )

    _, _, tc0 = thresholds(qp, 0)

    # Work on a full snapshot copy: every read below comes from the original
    # input, so writing q-side samples can never contaminate a p-side read.
    out = [list(row) for row in frame]

    for k in range(4):
        if vertical:
            row = y + k
            p = tuple(frame[row][x - 4 + i] for i in range(4))
            q = tuple(frame[row][x + i] for i in range(4))
        else:
            col = x + k
            p = tuple(frame[y - 4 + i][col] for i in range(4))
            q = tuple(frame[y + i][col] for i in range(4))

        bS = boundary_strength(p, q, qp, intra=False)
        new_p, new_q = filter_luma(p, q, bS, tc0)

        if vertical:
            row = y + k
            for i in range(4):
                out[row][x - 4 + i] = new_p[i]
                out[row][x + i] = new_q[i]
        else:
            col = x + k
            for i in range(4):
                out[y - 4 + i][col] = new_p[i]
                out[y + i][col] = new_q[i]

    return out


def filter_frame(frame: Frame, qp_map: QPMap, block: int) -> list[list[int]]:
    """Deblock every internal block edge of one frame and return a new frame.

    ``block`` is the block size.  Vertical boundaries are at
    ``x = block, 2*block, ... < width`` and horizontal boundaries at
    ``y = block, 2*block, ... < height``.  Each boundary is split into
    four-sample segments.  Each unique segment is passed to
    :func:`filter_edge` exactly once; the QP is read from ``qp_map`` at the
    segment's first q sample.

    Reads for every candidate segment come from the same input snapshot.
    When one pixel lies in the support of both a vertical and a horizontal
    edge, the result of the geometrically nearest boundary is used (on a tie
    the vertical result wins); the two filtered values are never applied one
    after the other.  Consequently the frame is not filtered twice and the
    result does not depend on the traversal order.

    ``frame`` and ``qp_map`` must have equal rectangular dimensions, every
    pixel must be in ``[0, 255]``, every QP in ``[0, 51]``, and ``block`` must
    be a positive multiple of four that divides both dimensions.  Invalid
    input raises ``ValueError`` with the messages from
    :func:`validate_frame`.
    """
    errors = validate_frame(frame, qp_map, block)
    if errors:
        raise ValueError("; ".join(errors))

    height = len(frame)
    width = len(frame[0])

    # Every candidate segment is filtered independently from the same input
    # snapshot.  Their outputs are merged afterwards, so no pixel is ever
    # filtered twice (no chained vertical-then-horizontal application).
    candidates: list[tuple[int, int, bool, Frame]] = []

    for bx in range(block, width, block):
        for y in range(0, height, 4):
            segment = filter_edge(frame, bx, y, True, qp_map[y][bx])
            candidates.append((bx, y, True, segment))

    for by in range(block, height, block):
        for x in range(0, width, 4):
            segment = filter_edge(frame, x, by, False, qp_map[by][x])
            candidates.append((x, by, False, segment))

    result = [list(row) for row in frame]
    # owner[y][x] = (distance, vertical_priority) of the currently winning
    # candidate for that pixel; ties resolve in favour of the vertical edge.
    owner: list[list[tuple[int, int] | None]] = [
        [None] * width for _ in range(height)
    ]

    for boundary_x, start_y, vertical, segment in candidates:
        if vertical:
            for row in range(start_y, start_y + 4):
                for col in range(boundary_x - 4, boundary_x + 4):
                    distance = abs(col - boundary_x)
                    key = (distance, 0)  # 0: vertical wins a tie
                    current = owner[row][col]
                    if current is None or key < current:
                        owner[row][col] = key
                        result[row][col] = segment[row][col]
        else:
            boundary_y = start_y
            for row in range(boundary_y - 4, boundary_y + 4):
                for col in range(boundary_x, boundary_x + 4):
                    distance = abs(row - boundary_y)
                    key = (distance, 1)  # 1: horizontal loses a tie
                    current = owner[row][col]
                    if current is None or key < current:
                        owner[row][col] = key
                        result[row][col] = segment[row][col]

    return result


def edge_map(frame: Frame, block: int) -> list[list[int]]:
    """Return a debug map of the internal block-boundary pixels.

    The result has the same shape as ``frame``.  A pixel is marked ``1`` when
    its column or row is a positive internal multiple of ``block`` (that is,
    ``x % block == 0`` with ``x > 0``, or the analogous condition for ``y``);
    all other pixels are ``0``.  Frame border coordinates are not internal
    boundaries and stay unmarked.

    This is a purely geometric map: pixel contrast must not create or remove
    an entry.  A strong image edge that is not block-aligned is not a block
    boundary.
    """
    height = len(frame)
    width = len(frame[0]) if height else 0
    return [
        [
            1 if (0 < x < width and x % block == 0)
            or (0 < y < height and y % block == 0)
            else 0
            for x in range(width)
        ]
        for y in range(height)
    ]


def validate_frame(frame: Frame, qp_map: QPMap, block: int) -> list[str]:
    """Return human-readable validation errors for a deblocking job.

    An empty list means the job is valid.  Error strings are lowercase and
    mention the relevant field.  The checks cover:

    * non-empty rectangular frame with every luma sample in ``[0, 255]``;
    * ``qp_map`` has exactly the frame shape and every QP in ``[0, 51]``;
    * ``block`` is a positive multiple of four;
    * ``block`` divides both frame dimensions  --  otherwise an internal
      edge segment would start outside the frame.

    Repeated or multiple independent failures may produce several strings.
    """
    errors: list[str] = []

    if not frame or not isinstance(frame, list) or any(
        not isinstance(row, list) for row in frame
    ):
        errors.append("frame must be a non-empty rectangular list of rows")
        return errors

    height = len(frame)
    width = len(frame[0])

    if width == 0 or any(len(row) != width for row in frame):
        errors.append("frame must be a non-empty rectangular list of rows")

    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            if not isinstance(value, int) or value < 0 or value > 255:
                errors.append(
                    f"frame pixel at (x={x}, y={y}) must be in [0, 255]"
                )

    if (
        not isinstance(qp_map, list)
        or len(qp_map) != height
        or any(not isinstance(row, list) or len(row) != width for row in qp_map)
    ):
        errors.append("qp_map must have exactly the frame shape")
    else:
        for y, row in enumerate(qp_map):
            for x, value in enumerate(row):
                if not isinstance(value, int) or value < 0 or value > 51:
                    errors.append(
                        f"qp_map value at (x={x}, y={y}) must be in [0, 51]"
                    )

    if not isinstance(block, int) or block <= 0 or block % 4 != 0:
        errors.append("block must be a positive multiple of four")

    # The divisibility check is independent of the multiple-of-four check so
    # that a non-conforming block is still reported as not dividing the frame.
    if isinstance(block, int) and block > 0:
        if width % block != 0:
            errors.append(
                f"block {block} must divide frame width {width}"
            )
        if height % block != 0:
            errors.append(
                f"block {block} must divide frame height {height}"
            )

    return errors


def deblock_batch(frames: list[Frame], config: dict) -> dict:
    """Filter a list of frames independently and return a batch result.

    ``config`` is ``{"block": int, "qp_maps": list[QPMap]}``.  ``qp_maps``
    has exactly one map per frame, in the same order; the map for frame ``i``
    is used only for frame ``i``.  Each frame is processed by
    :func:`filter_frame`, and no mutable buffer is shared between frames.

    The return value is::

        {
            "frames": [filtered_frame, ...],
            "edge_maps": [edge_map, ...],
            "count": len(frames),
        }

    Input order is preserved.  A length mismatch, an invalid block, or an
    invalid frame/QP map raises ``ValueError`` before any result is returned;
    the input frames are never mutated.
    """
    block = config["block"]
    qp_maps = config["qp_maps"]

    if len(frames) != len(qp_maps):
        raise ValueError(
            "qp_maps must contain exactly one qp map per frame"
        )

    # Validate every job up front so a bad later frame cannot surface after
    # partial results have been produced.
    for index, (frame, qp_map) in enumerate(zip(frames, qp_maps)):
        errors = validate_frame(frame, qp_map, block)
        if errors:
            raise ValueError(f"frame {index}: " + "; ".join(errors))

    filtered_frames: list[Frame] = []
    maps: list[list[list[int]]] = []
    # Each frame uses its own QP map and gets fresh buffers; nothing mutable
    # is carried over from one frame to the next.
    for frame, qp_map in zip(frames, qp_maps):
        filtered_frames.append(filter_frame(frame, qp_map, block))
        maps.append(edge_map(frame, block))

    return {
        "frames": filtered_frames,
        "edge_maps": maps,
        "count": len(frames),
    }
