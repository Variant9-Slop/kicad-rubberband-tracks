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
from typing import Iterable, Sequence

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
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
