"""Pure-geometry core: pull a polyline taut around obstacles without changing
which side of each obstacle it passes on.

Method:
  1. Each obstacle is grown by its required distance, but never by more than the
     original path's own distance to it. The new path is never tighter than the
     rules, and never tighter than the original was; a gap the original fits
     through stays open.
  2. Free space = convex hull of the path (plus margin) minus the grown obstacles.
  3. Triangulate the free space (constrained Delaunay).
  4. Walk the original path to get the strip ("sleeve") of triangles it crosses.
  5. Funnel algorithm through that strip = shortest path with the same topology.

Coordinates are plain floats (the caller uses nanometres).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, NamedTuple, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree

Pt = tuple[float, float]

QUAD_SEGS = 32
# GEOS may use arc steps up to ~1.5x the nominal pi/(2*QUAD_SEGS); measured worst inward
# error at 32 segments is 0.068 % of the radius. Scale for a full nominal step (0.12 %).
_BUF_SCALE = 1.0 / math.cos(math.pi / (2 * QUAD_SEGS))


class RubberBandError(Exception):
    def __init__(self, msg: str, where: Pt | None = None):
        super().__init__(msg)
        self.where = where


# ---------------------------------------------------------------- helpers

def _cross(o: Pt, a: Pt, b: Pt) -> float:
    """> 0 when b is to the left of the ray o->a."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def path_length(pts: Sequence[Pt]) -> float:
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def simplify_collinear(pts: Sequence[Pt], tol: float = 1.0) -> list[Pt]:
    """Drop duplicate points and points lying on the line between their neighbours."""
    out: list[Pt] = []
    for p in pts:
        if out and math.dist(out[-1], p) < tol:
            continue
        while len(out) >= 2:
            a, b = out[-2], out[-1]
            seg = math.dist(a, p)
            if seg > 0 and abs(_cross(a, p, b)) / seg < tol and \
                    (b[0] - a[0]) * (p[0] - b[0]) + (b[1] - a[1]) * (p[1] - b[1]) >= 0:
                out.pop()
            else:
                break
        out.append(p)
    if len(pts) and out[-1] != tuple(pts[-1]):
        out[-1] = tuple(pts[-1])        # always keep the exact end point
    return out


# ---------------------------------------------------------------- triangulation / sleeve

def _tri_coords_all(tri_geoms) -> list[list[Pt]]:
    xy = shapely.get_coordinates(np.asarray(tri_geoms, dtype=object)).reshape(-1, 4, 2)[:, :3]
    return [[(float(a), float(b)) for a, b in t] for t in xy.tolist()]


def _build_adjacency(tris: list[list[Pt]]) -> list[list[int]]:
    edge_owner: dict[tuple[Pt, Pt], int] = {}
    adj: list[list[int]] = [[] for _ in tris]
    for i, t in enumerate(tris):
        for k in range(3):
            a, b = t[k], t[(k + 1) % 3]
            key = (a, b) if a < b else (b, a)
            j = edge_owner.get(key)
            if j is None:
                edge_owner[key] = i
            else:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _bfs(adj: list[list[int]], src: int, dst: int) -> list[int]:
    prev = {src: -1}
    q = deque([src])
    while q:
        n = q.popleft()
        if n == dst:
            break
        for m in adj[n]:
            if m not in prev:
                prev[m] = n
                q.append(m)
    if dst not in prev:
        raise RubberBandError("path crosses an obstacle (no connected free space)")
    out = []
    n = dst
    while n != -1:
        out.append(n)
        n = prev[n]
    return out[::-1]


def _sleeve(path: Sequence[Pt], tri_geoms: list[Polygon], adj) -> list[int]:
    """Triangles crossed by the path, in order, without loops."""
    tree = STRtree(tri_geoms)
    tri_arr = np.asarray(tri_geoms, dtype=object)
    seq: list[int] = []

    def push(idx: int):
        if not seq:
            seq.append(idx)
        elif idx != seq[-1]:
            if idx in adj[seq[-1]]:
                seq.append(idx)
            else:   # path passes exactly through a vertex: bridge locally
                seq.extend(_bfs(adj, seq[-1], idx)[1:])

    start = Point(path[0])
    first = [int(i) for i in tree.query(start, predicate="intersects")]
    push(first[0] if first else int(tree.query_nearest(start)[0]))

    for a, b in zip(path, path[1:]):
        seg = LineString([a, b])
        L = seg.length
        if L == 0:
            continue
        cand = tree.query(seg, predicate="intersects")
        if len(cand) == 0:
            continue
        inter = shapely.intersection(tri_arr[cand], seg)
        keep = shapely.length(inter) >= 1.0       # drop touches at a vertex / slivers
        cand, inter = cand[keep], inter[keep]
        if len(cand) == 0:
            continue
        xy, idx = shapely.get_coordinates(inter, return_index=True)
        t = shapely.line_locate_point(seg, shapely.points(xy))
        t_min = np.full(len(cand), np.inf)
        t_max = np.full(len(cand), -np.inf)
        np.minimum.at(t_min, idx, t)
        np.maximum.at(t_max, idx, t)
        for k in np.lexsort((t_max, t_min)):
            push(int(cand[k]))

    end = Point(path[-1])
    last = [int(i) for i in tree.query(end, predicate="intersects")]
    if last and seq[-1] not in last:
        push(last[0])

    # Cancel immediate back-and-forth (X, Y, X -> X). The triangulation has no interior
    # vertices, so the only real loops in the triangle graph go around obstacles; a
    # walk with no back-and-forth is therefore the unique representative of the path's
    # topology. (Removing every repeated triangle would wrongly cut out detours around
    # obstacles.)
    out: list[int] = []
    for t in seq:
        if len(out) >= 2 and out[-2] == t:
            out.pop()
        elif not out or out[-1] != t:
            out.append(t)

    # An endpoint can lie exactly on a triangle edge (a via in line with the tips of two
    # grown obstacles), so the triangle picked for it may be one the path only touches.
    # The portal into the next triangle would then pass through the funnel apex, which
    # opens the funnel to 180° and makes it wrap a neighbouring obstacle. Start and end
    # in the triangles the path actually runs through.
    while len(out) >= 2 and tri_geoms[out[1]].distance(start) < 1.0:
        out.pop(0)
    while len(out) >= 2 and tri_geoms[out[-2]].distance(end) < 1.0:
        out.pop()
    return out


# ---------------------------------------------------------------- funnel

def _portals(sleeve: list[int], tris: list[list[Pt]]) -> list[tuple[Pt, Pt]]:
    """Shared edges between consecutive triangles, as (left, right) seen in travel direction."""
    portals = []
    for i in range(len(sleeve) - 1):
        a_t, b_t = tris[sleeve[i]], tris[sleeve[i + 1]]
        shared = [v for v in a_t if v in b_t]
        if len(shared) != 2:
            raise RubberBandError("internal: sleeve triangles not adjacent")
        c = next(v for v in a_t if v not in shared)
        a, b = shared
        portals.append((b, a) if _cross(c, a, b) > 0 else (a, b))
    return portals


def _funnel(start: Pt, end: Pt, portals: list[tuple[Pt, Pt]]) -> list[Pt]:
    ps = [(start, start)] + portals + [(end, end)]
    path = [start]
    apex = left = right = start
    apex_i = left_i = right_i = 0
    i = 1
    while i < len(ps):
        l, r = ps[i]
        if _cross(apex, right, r) >= 0:             # tighten right side
            if apex == right or _cross(apex, left, r) < 0:
                right, right_i = r, i
            else:
                path.append(left)
                apex = right = left
                apex_i = right_i = left_i
                i = apex_i + 1
                continue
        if _cross(apex, left, l) <= 0:              # tighten left side
            if apex == left or _cross(apex, right, l) > 0:
                left, left_i = l, i
            else:
                path.append(right)
                apex = left = right
                apex_i = left_i = right_i
                i = apex_i + 1
                continue
        i += 1
    if path[-1] != end:
        path.append(end)
    return path


def _nudge_out(path: Sequence[Pt], blocked) -> list[Pt]:
    """Move interior vertices that sit a hair inside an obstacle (rounding of a path that
    already hugs it) to the nearest free spot a few nm away. Only used to trace the
    sleeve, so the path's topology is read from where it really runs."""
    if blocked.is_empty or len(path) < 3:
        return list(path)
    pts = np.asarray(path[1:-1], dtype=float)
    inside = shapely.contains_xy(blocked, pts[:, 0], pts[:, 1])
    if not inside.any():
        return list(path)
    out = list(path)
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    for k in np.nonzero(inside)[0]:
        x, y = pts[k]
        for rad in (5.0, 10.0, 20.0, 50.0, 100.0, 250.0):
            cx, cy = x + rad * np.cos(angles), y + rad * np.sin(angles)
            free = ~shapely.contains_xy(blocked, cx, cy)
            if free.any():
                j = int(np.argmax(free))
                out[k + 1] = (float(cx[j]), float(cy[j]))
                break
    return out


# ---------------------------------------------------------------- public API

def pull_taut(path: Sequence[Pt], obstacles: Iterable[tuple], eps: float = 2.0,
              hull_margin: float = 1_000_000.0) -> list[Pt]:
    """Shortest path homotopic to `path` that keeps clear of `obstacles`.

    obstacles: (geometry, required_distance[, reference_distance]) tuples. The path's
    centreline must stay `required_distance` away from the geometry, relaxed to
    `reference_distance` (default: this path's current distance) minus `eps` where that
    is closer. Pass the original track's distance as reference when calling repeatedly,
    so later passes can't creep closer than the original was.
    Endpoints stay fixed.
    """
    path = simplify_collinear(path)
    if len(path) < 3:
        return list(path)
    start, end = path[0], path[-1]
    orig = LineString(path)

    hull = orig.convex_hull.buffer(hull_margin, join_style="mitre")
    obstacles = list(obstacles)
    if obstacles:
        ob_geoms = np.asarray([ob[0] for ob in obstacles], dtype=object)
        d = np.asarray([ob[1] for ob in obstacles], dtype=float)
        near = shapely.distance(hull, ob_geoms) <= d
        ob_geoms, d = ob_geoms[near], d[near]
        refs = [obstacles[i][2] if len(obstacles[i]) > 2 else None for i in np.nonzero(near)[0]]
    else:
        ob_geoms, d, refs = np.empty(0, dtype=object), np.empty(0), []
    if len(ob_geoms):
        dist_now = shapely.distance(orig, ob_geoms)
        ref = np.asarray([dn if r is None else r for dn, r in zip(dist_now, refs)], dtype=float)
        if (ref <= eps).any():
            g = ob_geoms[int(np.argmin(ref))]
            p = shapely.ops.nearest_points(orig, g)[0]
            raise RubberBandError("original track touches other copper", (p.x, p.y))
        # polygon buffer circumscribes the true offset; capped where the track was already closer
        r = np.minimum(d * _BUF_SCALE, ref - eps)
        grown = shapely.buffer(ob_geoms, r, quad_segs=QUAD_SEGS)
        check_dist = r / _BUF_SCALE - 5.0
        blocked = shapely.union_all(grown)
    else:
        blocked = Polygon()
    free = hull.difference(blocked)
    if free.is_empty:
        raise RubberBandError("no free space around the path", start)
    if isinstance(free, MultiPolygon):
        parts = np.asarray(list(free.geoms), dtype=object)
        # the region the original path runs through (it may only graze it at the start)
        d_part = shapely.distance(parts, orig.interpolate(0.5, normalized=True)) + \
            shapely.distance(parts, Point(start)) + shapely.distance(parts, Point(end))
        free = parts[int(np.argmin(d_part))]

    tri_geoms = list(shapely.constrained_delaunay_triangles(free).geoms)
    # guard against triangles a degenerate input might put inside an obstacle
    shapely.prepare(free)
    inside = shapely.covers(free, shapely.centroid(np.asarray(tri_geoms, dtype=object)))
    tri_geoms = [t for t, ok in zip(tri_geoms, inside) if ok]
    if not tri_geoms:
        raise RubberBandError("triangulation failed", start)
    tris = _tri_coords_all(tri_geoms)
    adj = _build_adjacency(tris)

    sleeve = _sleeve(_nudge_out(path, blocked), tri_geoms, adj)
    result = simplify_collinear(_funnel(start, end, _portals(sleeve, tris)))

    new = LineString(result)
    if len(ob_geoms):
        bad = shapely.distance(new, ob_geoms) < check_dist
        if bad.any():
            p = shapely.ops.nearest_points(new, ob_geoms[int(np.argmax(bad))])[0]
            raise RubberBandError("result would violate clearance", (p.x, p.y))
    if path_length(result) > path_length(path) + 1.0:
        raise RubberBandError("internal: result is longer than the original", start)
    return result


# ---------------------------------------------------------------- exact path
#
# A shortest path that keeps clear of obstacles is made of straight lines and arcs round
# obstacle features (a via or hole centre, a pad or polygon corner, a track end, an arc's
# centre) at the required distance. pull_taut finds which features the path wraps, on
# polygons; exact_path then builds the true lines and arcs round those features and checks
# them against the true shapes. Nothing is fitted or guessed.

ARC_MIN_SAGITTA = 3.0   # flatter arcs (nm off their chord) become a corner: rounding would warp them
SNAP_TOL = 2_000.0      # an original this close to the required distance counts as meant to be at it
ROUNDING = 2.0          # added to exact distances: the caller rounds to whole nm
CHECK_ERR = 0.1         # curves are checked as chords this close to them
CHECK_TOL = 5.0         # allowed shortfall (nm): other tracks are on whole nm, their pieces
                        # don't quite meet; KiCad's DRC ignores far more (its epsilon, 0.5 um)


class Ob(NamedTuple):
    """An obstacle: `geom` must stay `d` away from the path.

    ref: the original track's distance to it (default: the path's current distance).
    Where that is closer than `d` (by more than SNAP_TOL) the path may come as close.
    arc: (cx, cy, r, start angle, signed sweep) when geom stands in for a circular arc
    (see arc_band); the path then wraps the arc's centre and is checked against the arc.
    """
    geom: object
    d: float
    ref: float | None = None
    arc: tuple | None = None


def arc_points(arc, max_err: float, side: str | None = None) -> list[Pt]:
    """Points of `arc` = (cx, cy, r, a0, sweep): chords within max_err inside it; with side
    "outer", a polyline along tangents just outside it; "inner", chords whose corners sit
    under the outer one's corners."""
    cx, cy, r, a0, sweep = arc
    step = 2 * math.acos(max(-1.0, 1 - max_err / r)) if r > max_err else math.pi / 8
    n = max(2, int(math.ceil(abs(sweep) / step)))
    if side is None:
        angles, rr = [a0 + sweep * i / n for i in range(n + 1)], r
    else:   # outer: corners where the tangents at a0 + sweep * i / n meet
        angles = [a0 + sweep * (i + 0.5) / n for i in range(n)]
        rr = r / math.cos(sweep / n / 2) if side == "outer" else r
    pts = [(cx + rr * math.cos(a), cy + rr * math.sin(a)) for a in angles]
    if side is not None:
        ends = [(cx + r * math.cos(a0), cy + r * math.sin(a0)),
                (cx + r * math.cos(a0 + sweep), cy + r * math.sin(a0 + sweep))]
        pts = [ends[0]] + pts + [ends[1]]
    return pts


def arc_band(arc, max_err: float):
    """An arc as a polygon: the thin band between chords inside it and tangents outside
    it, so keeping clear of the band keeps clear of the arc from either side."""
    outer = arc_points(arc, max_err, "outer")
    band = Polygon(outer + arc_points(arc, max_err, "inner")[-2:0:-1])
    return band if band.is_valid else LineString(outer)


def _circumcentre(a: Pt, b: Pt, c: Pt) -> Pt | None:
    d = 2 * _cross(a, b, c)
    if abs(d) < 1e-9:
        return None
    a2 = a[0] ** 2 + a[1] ** 2
    b2 = b[0] ** 2 + b[1] ** 2
    c2 = c[0] ** 2 + c[1] ** 2
    ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d
    uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d
    return (ux, uy)


def sample_elements(elements, max_err: float = 1.0) -> list[Pt]:
    """Polyline through `elements` ("line", a, b) / ("arc", a, mid, b), arcs split into
    chords at most `max_err` off the arc."""
    out: list[Pt] = []
    for e in elements:
        a, b = e[1], e[-1]
        if not out:
            out.append(a)
        c = _circumcentre(*e[1:]) if e[0] == "arc" else None
        if c is None:
            out.append(b)
            continue
        r = math.dist(c, a)
        a0 = math.atan2(a[1] - c[1], a[0] - c[0])
        sweep = (math.atan2(b[1] - c[1], b[0] - c[0]) - a0) % (2 * math.pi)
        if (math.atan2(e[2][1] - c[1], e[2][0] - c[0]) - a0) % (2 * math.pi) > sweep:
            sweep -= 2 * math.pi            # mid not on the counter-clockwise sweep
        step = 2 * math.acos(max(-1.0, 1 - max_err / r)) if r > max_err else math.pi / 8
        n = max(2, int(math.ceil(abs(sweep) / step)))
        out.extend((c[0] + r * math.cos(a0 + sweep * i / n), c[1] + r * math.sin(a0 + sweep * i / n))
                   for i in range(1, n))
        out.append(b)
    return out


def _tangent(ca, ra: float, cb, rb: float) -> tuple[np.ndarray, np.ndarray]:
    """Line leaving circle a and reaching circle b. Radii are signed: > 0 when the path
    turns left (counter-clockwise) around that circle. Raises _NoTangent if there is none."""
    dx, dy = cb - ca
    L = math.hypot(dx, dy)
    k = rb - ra
    if L <= abs(k) + TOUCH_TOL:
        if abs(k) - L > 200.0 or L < 1.0:
            raise _NoTangent()
        # The circles touch (an arc track and the segment it runs into), or all but: go
        # from one straight onto the other there. A tangent across a gap of a few nm
        # between big circles would be a sliver microns long.
        u = np.array([dx, dy]) / L
        if ra * rb > 0 and L > abs(k):
            # one pokes out of the other by a few nm (pieces rounded to whole nm don't
            # quite meet): switch where they cross, so the path stays outside both
            p = _crossing(ca, ra, cb, rb)
            if p is not None:
                return p, p
        p = cb - abs(rb) * u if ra * rb > 0 and abs(rb) > abs(ra) else ca + abs(ra) * u
        return p, p
    th = math.atan2(dy, dx) - math.asin(k / L)
    n = np.array([-math.sin(th), math.cos(th)])
    return ca - ra * n, cb - rb * n


TOUCH_TOL = 10.0        # circles closer than this to touching are taken as touching


class _NoTangent(Exception):
    pass


def _crossing(ca, ra: float, cb, rb: float):
    """Where a path going round circle a (signed radius ra) runs into circle b: of the two
    points where they cross, the one where carrying on round a would enter b."""
    d = cb - ca
    L = math.hypot(*d)
    x = (L * L + ra * ra - rb * rb) / (2 * L)
    h2 = ra * ra - x * x
    if h2 < 0:
        return None
    u, h = d / L, math.sqrt(h2)
    n = np.array([-u[1], u[0]])
    for p in (ca + x * u + h * n, ca + x * u - h * n):
        r = p - ca
        along = math.copysign(1, ra) * np.array([-r[1], r[0]])     # direction round a
        if np.dot(along, cb - p) > 0:
            return p
    return None


def _turn(a: Pt, b: Pt, c: Pt) -> float:
    """Signed turning angle at b (> 0 = left)."""
    u = (b[0] - a[0], b[1] - a[1])
    v = (c[0] - b[0], c[1] - b[1])
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


class _Wrap(NamedTuple):
    centre: np.ndarray
    radius: float           # > 0
    sign: float             # +1: path turns left around it
    turn: float | None      # turning of the pull_taut corners it replaces, if any


def exact_path(path: Sequence[Pt], obstacles: Iterable[tuple], tol: float = CHECK_TOL,
               min_sag: float = ARC_MIN_SAGITTA, max_rounds: int = 40) -> list[tuple]:
    """See _exact_path. Arcs get ROUNDING to spare on their radius (the caller rounds to
    whole nm, and neighbouring pieces of rounded tracks don't quite meet); without it if
    that doesn't fit (a squeeze at exactly the clearance on both sides)."""
    try:
        return _exact_path(path, obstacles, ROUNDING, tol, min_sag, max_rounds)
    except RubberBandError:
        return _exact_path(path, obstacles, 0.0, tol, min_sag, max_rounds)


def _exact_path(path, obstacles, spare, tol, min_sag, max_rounds) -> list[tuple]:
    """Exact lines and arcs for a taut path from pull_taut (same obstacles).

    Each corner of the taut path lies where pull_taut's polygons round a feature of an
    obstacle; consecutive corners on one feature become one wrap round it, at the exact
    distance. Wraps are joined by exact tangents. The result is checked against the true
    shapes: a feature it cuts into (the polygons missed it) is added as a wrap, a wrap it
    no longer touches is dropped, until it is clear.

    Returns [("line", a, b) | ("arc", a, mid, b)] from path[0] to path[-1]; arcs too flat
    to survive rounding (under min_sag) become a corner where the tangents meet.
    Raises RubberBandError when no clean path is found.
    """
    path = simplify_collinear(path)
    start, end = path[0], path[-1]
    if len(path) < 3:
        return [("line", start, end)]
    line = LineString(path)
    obs = [o if isinstance(o, Ob) else Ob(*o) for o in obstacles]
    if obs:
        geoms = np.asarray([o.geom for o in obs], dtype=object)
        d = np.asarray([o.d for o in obs], dtype=float)
        near = shapely.distance(line, geoms) <= d * _BUF_SCALE * 1.01 + SNAP_TOL + 10_000.0
        obs = [o for o, n in zip(obs, near) if n]
    geoms = np.asarray([o.geom for o in obs], dtype=object)
    d = np.asarray([o.d for o in obs], dtype=float)
    # true shapes for the clearance check: arcs as arcs, not the band standing in for them
    check = np.asarray([LineString(arc_points(o.arc, CHECK_ERR)) if o.arc else o.geom for o in obs], dtype=object)
    ref = np.asarray([o.ref if o.ref is not None else line.distance(c) for o, c in zip(obs, check)], dtype=float)
    # never closer than the rules, or than the original where that was (clearly) closer;
    # and no more than the fixed ends have, which can't move
    ends = shapely.distance(shapely.points([start, end])[:, None], check[None, :]).min(axis=0) \
        if len(obs) else ref
    keep = np.minimum(np.where(ref >= d - SNAP_TOL, d, ref), ends)

    def feature(k, v):
        """(centre, radius) of the feature of obstacle k that point v wraps, or None."""
        o = obs[k]
        if o.arc is not None:
            cx, cy, r, a0, sweep = o.arc
            t = (math.atan2(v[1] - cy, v[0] - cx) - a0) * math.copysign(1, sweep) % (2 * math.pi)
            if t <= abs(sweep) and math.hypot(v[0] - cx, v[1] - cy) > r:
                return np.array([cx, cy]), r + keep[k] + spare
            tips = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (a0, a0 + sweep)]
            return np.asarray(min(tips, key=lambda e: math.dist(e, v))), keep[k] + spare
        g = o.geom
        if g.geom_type == "Point":
            return np.array([g.x, g.y]), keep[k] + spare
        p = shapely.ops.nearest_points(g, Point(v))[0]
        verts = shapely.get_coordinates(g)
        j = int(np.argmin(np.hypot(verts[:, 0] - p.x, verts[:, 1] - p.y)))
        if math.hypot(verts[j, 0] - p.x, verts[j, 1] - p.y) > 1.0:
            return None         # on a straight side: nothing to wrap
        return verts[j].copy(), keep[k] + spare

    # the feature each corner wraps: on an obstacle's grown outline, inside the bend
    inner = path[1:-1]
    turns = [_turn(path[i], path[i + 1], path[i + 2]) for i in range(len(inner))]
    D = shapely.distance(shapely.points(inner)[:, None], geoms[None, :]) if len(obs) else None
    wraps: list[_Wrap] = []
    for i, v in enumerate(inner):
        if D is None:
            break
        a, b = np.subtract(path[i], v), np.subtract(path[i + 2], v)
        inward = a / np.hypot(*a) + b / np.hypot(*b)
        for k in np.argsort(D[i] / d):
            k = int(k)
            if D[i, k] > d[k] * _BUF_SCALE + 2.0:
                break
            f = feature(k, v)
            if f is None or np.dot(f[0] - v, inward) <= 0:
                continue
            s = math.copysign(1.0, turns[i])
            last = wraps[-1] if wraps else None
            if last and last.sign == s and math.dist(last.centre, f[0]) <= 1.0 and abs(last.radius - f[1]) <= 1.0:
                wraps[-1] = last._replace(turn=last.turn + turns[i])
            else:
                wraps.append(_Wrap(f[0], f[1], s, turns[i]))
            break

    for _ in range(max_rounds):
        try:
            elements, owner = _build(start, end, wraps, min_sag)
        except _Drop as e:
            del wraps[e.wrap]
            continue
        if len(obs):
            pts = sample_elements(elements, CHECK_ERR)
            curve = LineString(pts)
            short = keep - tol - shapely.distance(curve, check)
            if short.max() > 0:
                k = int(np.argmax(short))
                p = shapely.ops.nearest_points(curve, check[k])[0]
                f = feature(k, (p.x, p.y))
                if f is None:
                    raise RubberBandError("would cut into an obstacle side", (p.x, p.y))
                # which side, and between which wraps
                xy = np.asarray(pts)
                i = int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
                q0, q1 = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
                s = 1.0 if _cross(q0, q1, tuple(f[0])) > 0 else -1.0
                el = min(range(len(elements)), key=lambda j: LineString(sample_elements([elements[j]], 10.0))
                         .distance(p))
                kind, j = owner[el]
                after = math.dist(elements[el][-1], (p.x, p.y)) < math.dist(elements[el][1], (p.x, p.y))
                at = j if kind == "gap" else j + after
                new = _Wrap(f[0], f[1], s, None)
                if any(math.dist(w.centre, new.centre) <= 1.0 and abs(w.radius - new.radius) <= 1.0 and w.sign == s
                       for w in wraps[max(0, at - 1):at + 1]):
                    raise RubberBandError("would cut into an obstacle", (p.x, p.y))
                wraps.insert(at, new)
                continue
        return elements
    raise RubberBandError("no clean path found", start)


class _Drop(Exception):
    def __init__(self, wrap: int):
        self.wrap = wrap


def _build(start, end, wraps, min_sag):
    """Lines and arcs through `wraps`, and for each piece what it belongs to: ("gap", j)
    = between wraps j - 1 and j (j = index a new wrap there would get), ("wrap", j) =
    round wraps[j]. Raises _Drop for a wrap the path no longer touches."""
    nodes = [(np.asarray(start, dtype=float), 0.0)] + [(w.centre, w.sign * w.radius) for w in wraps] + \
        [(np.asarray(end, dtype=float), 0.0)]
    tangents = []
    for j in range(len(nodes) - 1):
        (ca, ra), (cb, rb) = nodes[j], nodes[j + 1]
        try:
            tangents.append(_tangent(ca, ra, cb, rb))
        except _NoTangent:
            # one circle inside the other (same way round): the inner one is not touched
            if ra * rb > 0:
                raise _Drop(j - 1 if abs(ra) < abs(rb) else j) from None
            if ra == 0 or rb == 0:      # an end inside a wrap
                raise _Drop(j if ra == 0 else j - 1) from None
            raise RubberBandError("no room between two obstacles", tuple(ca)) from None
    pt = lambda p: (float(p[0]), float(p[1]))
    elements, owner = [("line", start, pt(tangents[0][1]))], [("gap", 0)]
    for j, w in enumerate(wraps):
        q_in, q_out = tangents[j][1], tangents[j + 1][0]
        a_in = math.atan2(*(q_in - w.centre)[::-1])
        sweep = (w.sign * (math.atan2(*(q_out - w.centre)[::-1]) - a_in)) % (2 * math.pi)
        if w.turn is not None:      # full loops round it, as the taut path made them
            sweep += 2 * math.pi * round((w.sign * w.turn - sweep) / (2 * math.pi))
        elif sweep > math.pi:
            sweep -= 2 * math.pi
        if sweep < -1e-9:
            raise _Drop(j)
        R = w.radius
        if R * (1 - math.cos(sweep / 2)) >= min_sag:
            n = int(math.ceil(sweep / math.pi - 1e-9))     # KiCad arcs: keep each under a half turn
            on = lambda a: (float(w.centre[0] + R * math.cos(a)), float(w.centre[1] + R * math.sin(a)))
            ends = [pt(q_in)] + [on(a_in + w.sign * sweep * m / n) for m in range(1, n)] + [pt(q_out)]
            for m in range(n):
                elements.append(("arc", ends[m], on(a_in + w.sign * sweep * (m + 0.5) / n), ends[m + 1]))
                owner.append(("wrap", j))
        else:   # a corner where the tangents meet, a hair outside the arc
            am = a_in + w.sign * sweep / 2
            q_out = w.centre + R / math.cos(sweep / 2) * np.array([math.cos(am), math.sin(am)])
            elements[-1] = (elements[-1][0], *elements[-1][1:-1], pt(q_out))
        nxt = end if j == len(wraps) - 1 else pt(tangents[j + 1][1])
        elements.append(("line", pt(q_out), nxt))
        owner.append(("gap", j + 1))
    # drop pieces of no length (touching circles), merge lines that are in line
    out, own = [], []
    for e, o in zip(elements, owner):
        if e[0] == "line" and math.dist(e[1], e[2]) < 0.5:
            continue
        if out and e[0] == "line" and out[-1][0] == "line":
            a, b, m = out[-1][1], e[2], e[1]
            if math.dist(a, b) > 0 and abs(_cross(a, b, m)) / math.dist(a, b) < min_sag:
                out[-1] = ("line", a, b)
                continue
        out.append(e)
        own.append(o)
    if not out:
        out, own = [("line", start, end)], [("gap", 0)]
    return out, own


def element_arc(e):
    """(cx, cy, r, a0, signed sweep) of an ("arc", a, mid, b) element, None if straight."""
    a, m, b = e[1:]
    c = _circumcentre(a, m, b)
    if c is None:
        return None
    a0 = math.atan2(a[1] - c[1], a[0] - c[0])
    sweep = (math.atan2(b[1] - c[1], b[0] - c[0]) - a0) % (2 * math.pi)
    if (math.atan2(m[1] - c[1], m[0] - c[0]) - a0) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    return c[0], c[1], math.dist(c, a), a0, sweep


def min_margin(elements, obstacles) -> float:
    """Smallest (distance - required) of `elements` to the true shapes of `obstacles`."""
    obs = [o if isinstance(o, Ob) else Ob(*o) for o in obstacles]
    if not obs:
        return math.inf
    curve = LineString(sample_elements(elements, CHECK_ERR))
    check = np.asarray([LineString(arc_points(o.arc, CHECK_ERR)) if o.arc else o.geom for o in obs], dtype=object)
    return float((shapely.distance(curve, check) - np.asarray([o.d for o in obs])).min())
