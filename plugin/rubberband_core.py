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


# ---------------------------------------------------------------- arcs

ARC_FIT_TOL = 10_000.0      # how far corner points may sit off one fitted circle (polygonised pads)
ARC_MIN_LENGTH = 10_000.0   # shorter arcs become a straight piece


def _fit_circle(pts) -> tuple[np.ndarray, float]:
    """Least-squares circle through `pts`: (centre, max distance of a point off the circle)."""
    a = np.asarray(pts, dtype=float)
    m = a.mean(axis=0)
    x = a - m
    A = np.column_stack([2 * x[:, 0], 2 * x[:, 1], np.ones(len(x))])
    sol = np.linalg.lstsq(A, (x ** 2).sum(axis=1), rcond=None)[0]
    c = sol[:2]
    r = math.sqrt(max(sol[2] + c @ c, 0.0))
    return c + m, float(np.abs(np.hypot(*(x - c).T) - r).max())


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
    turns left (counter-clockwise) around that circle."""
    dx, dy = cb - ca
    L = math.hypot(dx, dy)
    k = rb - ra
    if L <= abs(k) + 1e-6:
        if abs(k) - L > 200.0 or L < 1.0:
            raise RubberBandError("arcs: no tangent between neighbouring corners", tuple(ca))
        # the circles touch (an arc around a rounded corner, then one around the polygon
        # point it ends on): go from one straight onto the other where they touch
        u = np.array([dx, dy]) / L
        if ra * rb > 0 and abs(rb) > abs(ra):
            p = cb - abs(rb) * u
        else:
            p = ca + abs(ra) * u
        return p, p
    th = math.atan2(dy, dx) - math.asin(k / L)
    n = np.array([-math.sin(th), math.cos(th)])
    return ca - ra * n, cb - rb * n


def _turn(a: Pt, b: Pt, c: Pt) -> float:
    """Signed turning angle at b (> 0 = left)."""
    u = (b[0] - a[0], b[1] - a[1])
    v = (c[0] - b[0], c[1] - b[1])
    return math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])


def fit_arcs(path: Sequence[Pt], obstacles: Iterable[tuple], margin: float = 0.0,
             fit_tol: float = ARC_FIT_TOL, min_arc: float = ARC_MIN_LENGTH) -> list[tuple]:
    """Replace the corners of a taut path (from pull_taut) by arcs.

    Every corner of a taut path lies on a grown obstacle, where the growing rounded a
    point, a polygon corner or a track end. Consecutive corners around the same point
    (or around one circle, for pads KiCad hands over as polygons) become one arc around
    it, at the required distance plus `margin`, joined by exact tangents. Where the path
    was capped closer than required (tighter original) the arc keeps that distance.

    obstacles: as for pull_taut; a reference distance, if given, is ignored.
    Returns [("line", a, b) | ("arc", a, mid, b)] from path[0] to path[-1].
    Raises RubberBandError when the arcs would not fit; the caller keeps the segments.
    """
    path = simplify_collinear(path)
    if len(path) < 3:
        return [("line", path[0], path[-1])]
    line = LineString(path)
    obstacles = list(obstacles)
    geoms = np.asarray([ob[0] for ob in obstacles], dtype=object)
    d = np.asarray([ob[1] for ob in obstacles], dtype=float)
    if len(geoms):
        near = shapely.distance(line, geoms) <= d * _BUF_SCALE + fit_tol + margin + 1000.0
        geoms, d = geoms[near], d[near]

    # which obstacle each corner wraps: on the inside of the bend, on its grown outline
    # (closest relative to its distance; a neighbour on the outside can be as close)
    inner = path[1:-1]
    corners = []        # (obstacle index or -1, centre of the grown rounding, distance)
    D = shapely.distance(shapely.points(inner)[:, None], geoms[None, :]) if len(geoms) else None
    for i, v in enumerate(inner):
        corners.append((-1, v, 0.0))
        if D is None:
            continue
        a, b = np.subtract(path[i], v), np.subtract(path[i + 2], v)
        inward = a / np.hypot(*a) + b / np.hypot(*b)
        for k in np.argsort(D[i] / d):
            k = int(k)
            if D[i, k] > d[k] * _BUF_SCALE + 2.0:
                break
            c = shapely.ops.nearest_points(geoms[k], Point(v))[0]
            if (c.x - v[0]) * inward[0] + (c.y - v[1]) * inward[1] > 0:
                corners[-1] = (k, (c.x, c.y), float(D[i, k]))
                break
    turns = [_turn(path[i], path[i + 1], path[i + 2]) for i in range(len(inner))]

    def circle(idx):
        """(centre, radius) of one arc for corners idx[0]..idx[-1], None if there is none,
        and how many distinct obstacle points they wrap."""
        k, r_v = corners[idx[0]][0], corners[idx[0]][2]
        cs = [corners[idx[0]][1]]
        for i in idx[1:]:
            if math.dist(cs[-1], corners[i][1]) > 1.0:
                cs.append(corners[i][1])
        if len(cs) == 1:
            centre, rho = np.asarray(cs[0]), 0.0
        elif len(cs) >= 4:     # any three points are on some circle
            # the points must be a polygonised circle: on one circle (KiCad puts the vertices
            # on the arc, so only rounding is allowed), and close enough together that the
            # polygon stays within fit_tol of it (not a square's corners)
            centre, resid = _fit_circle(cs)
            rho = math.dist(centre, cs[0])
            ang = [math.atan2(c[1] - centre[1], c[0] - centre[0]) for c in cs]
            step = max(abs((b - a + math.pi) % (2 * math.pi) - math.pi) for a, b in zip(ang, ang[1:]))
            if resid > 100.0 or rho * (1 - math.cos(step / 2)) > fit_tol or \
                    any(abs(math.dist(centre, inner[i]) - rho - r_v) > fit_tol for i in idx):
                return None, len(cs)
        else:
            return None, len(cs)
        # r_v below the full grown size: pull_taut capped it at the original's distance
        a = d[k] + margin if r_v >= d[k] * _BUF_SCALE - 2.0 else r_v
        return (centre, rho + a), len(cs)

    # group corners greedily into runs that share obstacle, turn direction and circle
    groups = []
    i = 0
    while i < len(inner):
        k = corners[i][0]
        best, j = [i], i + 1
        while k >= 0 and j < len(inner) and corners[j][0] == k and turns[j] * turns[i] > 0:
            cand, n_points = circle(list(range(i, j + 1)))
            if cand is None:
                if n_points >= 4:   # fewer points may still become a circle with more
                    break
            else:
                best = list(range(i, j + 1))
            j += 1
        groups.append(best)
        i = best[-1] + 1

    def split(idx):
        """One group per obstacle point."""
        out = [[idx[0]]]
        for i in idx[1:]:
            if math.dist(corners[out[-1][-1]][1], corners[i][1]) <= 1.0:
                out[-1].append(i)
            else:
                out.append([i])
        return out

    # A fitted circle can still be wrong just past the points it was fitted to (a few
    # polygon points that happen to lie on one circle); then fall back to one arc per point.
    while True:
        try:
            elements = _arc_elements(path, inner, corners, turns, groups, circle, min_arc)
            if len(geoms):
                # never closer to anything than the rules, or than the segments already were
                curve = LineString(sample_elements(elements))
                limit = np.minimum(d, shapely.distance(line, geoms)) - 5.0
                bad = shapely.distance(curve, geoms) < limit
                if bad.any():
                    p = shapely.ops.nearest_points(curve, geoms[int(np.argmax(bad))])[0]
                    raise _ArcFailure("arcs would violate clearance", (p.x, p.y), range(len(groups)))
            return elements
        except _ArcFailure as e:
            redo = [g for g in e.groups if len(split(groups[g])) > 1]
            if not redo:
                raise RubberBandError(str(e), e.where) from None
            for g in sorted(redo, reverse=True):
                groups[g:g + 1] = split(groups[g])


class _ArcFailure(RubberBandError):
    def __init__(self, msg: str, where, groups):
        super().__init__(msg, where)
        self.groups = list(groups)


def _arc_elements(path, inner, corners, turns, groups, circle, min_arc) -> list[tuple]:
    """Lines and arcs for fit_arcs: one circle per group of corners, joined by tangents."""
    nodes = [(np.asarray(path[0], dtype=float), 0.0, None)]     # (centre, signed radius, corner idx)
    for idx in groups:
        if corners[idx[0]][0] < 0:      # corner not on any obstacle: keep it sharp
            nodes.append((np.asarray(inner[idx[0]], dtype=float), 0.0, idx))
        else:
            centre, rad = circle(idx)[0]
            nodes.append((centre, math.copysign(rad, turns[idx[0]]), idx))
    nodes.append((np.asarray(path[-1], dtype=float), 0.0, None))

    tangents = []
    for j in range(len(nodes) - 1):
        (ca, ra, _), (cb, rb, _) = nodes[j], nodes[j + 1]
        try:
            tangents.append(_tangent(ca, ra, cb, rb))
        except RubberBandError as e:
            raise _ArcFailure(str(e), e.where, [g for g in (j - 1, j) if 0 <= g < len(groups)]) from None
    pt = lambda p: (float(p[0]), float(p[1]))
    elements = [("line", path[0], pt(tangents[0][1]))]
    for j in range(1, len(nodes) - 1):
        centre, srad, idx = nodes[j]
        q_in, q_out = tangents[j - 1][1], tangents[j][0]
        if srad != 0:
            s = 1.0 if srad > 0 else -1.0
            a_in = math.atan2(*(q_in - centre)[::-1])
            sweep = (s * (math.atan2(*(q_out - centre)[::-1]) - a_in)) % (2 * math.pi)
            turn = s * sum(turns[i] for i in idx)
            sweep += 2 * math.pi * round((turn - sweep) / (2 * math.pi))    # full loops
            if abs(sweep - turn) > 0.25:
                raise _ArcFailure("arcs: arc does not match the corners it replaces", pt(centre), [j - 1])
            if sweep * abs(srad) >= min_arc:
                n = int(math.ceil(sweep / math.pi - 1e-9))     # KiCad arcs: keep each under a half turn
                on = lambda a: (float(centre[0] + abs(srad) * math.cos(a)), float(centre[1] + abs(srad) * math.sin(a)))
                ends = [pt(q_in)] + [on(a_in + s * sweep * m / n) for m in range(1, n)] + [pt(q_out)]
                for m in range(n):
                    elements.append(("arc", ends[m], on(a_in + s * sweep * (m + 0.5) / n), ends[m + 1]))
            else:
                elements.append(("line", pt(q_in), pt(q_out)))
        end = path[-1] if j == len(nodes) - 2 else pt(tangents[j][1])
        elements.append(("line", pt(q_out), end))
    if len(nodes) == 2:
        elements = [("line", path[0], path[-1])]
    return elements
