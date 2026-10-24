"""Geometry core: shortest path with fixed topology, clearance never reduced."""

import math
import random

import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from rubberband_core import RubberBandError, path_length, pull_taut

MM = 1e6


def mm(pts):
    return [(x * MM, y * MM) for x, y in pts]


def min_margin(path, obstacles):
    line = LineString(path)
    return min(line.distance(g) - d for g, d in obstacles)


def kicad_circle(cx, cy, r, err=5000):
    """Round pad as KiCad exports it: polygon with vertices on the circle."""
    n = max(8, int(math.ceil(math.pi / math.acos(1 - err / r))))
    return Polygon([(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
                    for i in range(n)])


@pytest.mark.parametrize("side", [1, -1])
def test_wraps_circle_on_original_side(side):
    obs = [(Point(5 * MM, 0), 1 * MM)]
    r = pull_taut(mm([(0, 0), (0, 3 * side), (10, 3 * side), (10, 0)]), obs)
    # analytic shortest path around a unit circle between (0,0) and (10,0)
    t = math.sqrt(5 ** 2 - 1)
    expected = 2 * t + 2 * math.asin(1 / 5)
    assert path_length(r) / MM == pytest.approx(expected, abs=0.005)
    y_mid = LineString(r).intersection(LineString([(5 * MM, -1e9), (5 * MM, 1e9)])).y
    assert math.copysign(1, y_mid) == side
    assert min_margin(r, obs) > -500


def test_free_path_becomes_straight():
    r = pull_taut(mm([(0, 0), (1, 2), (2, -2), (3, 2), (4, 0)]), [])
    assert r == mm([(0, 0), (4, 0)])


def test_keeps_longer_side_when_original_took_it():
    obs = [(Point(5 * MM, 0.5 * MM), 1 * MM)]
    r = pull_taut(mm([(0, 0), (0, -4), (10, -4), (10, 0)]), obs)
    y_mid = LineString(r).intersection(LineString([(5 * MM, -1e9), (5 * MM, 1e9)])).y
    assert y_mid < -0.4 * MM


def test_exact_width_gap_stays_open():
    """Track routed through a gap that fits it exactly (KiCad router at min clearance)."""
    d = 0.175 * MM
    pads = [box(x * MM, 0, x * MM + 0.3 * MM, 1.0 * MM) for x in (0, 0.65)]
    mid = (0.3 + 0.65) / 2
    path = mm([(mid - 1, -2), (mid - 1, -1), (mid, -0.5), (mid, 1.5), (mid + 1, 2), (mid + 1, 3)])
    obs = [(p, d) for p in pads]
    r = pull_taut(path, obs)
    assert path_length(r) < path_length(path)
    assert min_margin(r, obs) > -500
    gap = LineString([(mid * MM - d, 0.5 * MM), (mid * MM + d, 0.5 * MM)])
    assert LineString(r).intersects(gap)


def test_hugging_round_pad():
    r_pad, d = 0.4 * MM, 0.175 * MM
    rr = (r_pad + d) / MM / math.cos(math.pi / 8)
    octagon = sorted([(5 + rr * math.cos(a), rr * math.sin(a))
                      for a in [math.pi * k / 4 + math.pi / 8 for k in range(4)]])
    path = mm([(2, 0.1), (3, 1.2)] + octagon + [(7, 1.2), (8, 0.1)])
    obs = [(kicad_circle(5 * MM, 0, r_pad), d + 5000)]
    r = pull_taut(path, obs)
    assert path_length(r) < path_length(path)
    # the original sits 5 um inside the requirement; the result may match that, not exceed it
    assert min_margin(r, obs) > min_margin(path, obs) - 500


def test_original_touching_copper_is_reported():
    with pytest.raises(RubberBandError):
        pull_taut(mm([(0, 0), (5, 0), (10, 1)]), [(Point(5 * MM, 0), 0.5 * MM)])


def test_random_obstacle_fields():
    rnd = random.Random(1)
    for _ in range(80):
        obs = []
        for _ in range(rnd.randint(1, 20)):
            x, y = rnd.uniform(0, 20) * MM, rnd.uniform(0, 20) * MM
            if rnd.random() < .5:
                obs.append((Point(x, y), rnd.uniform(.2, 1.5) * MM))
            else:
                obs.append((box(x, y, x + rnd.uniform(.1, 3) * MM, y + rnd.uniform(.1, 3) * MM),
                            rnd.uniform(.05, .5) * MM))
        blocked = unary_union([g.buffer(d) for g, d in obs])
        pts, tries = [], 0
        while len(pts) < rnd.randint(3, 10) and tries < 2000:
            tries += 1
            p = (rnd.uniform(0, 20) * MM, rnd.uniform(0, 20) * MM)
            if blocked.contains(Point(p)) or blocked.distance(Point(p)) < 1000:
                continue
            if pts and LineString([pts[-1], p]).intersects(blocked):
                continue
            pts.append(p)
        if len(pts) < 3:
            continue
        r = pull_taut(pts, obs)
        assert r[0] == pts[0] and r[-1] == pts[-1]
        assert path_length(r) <= path_length(pts) + 1
        assert min_margin(r, obs) > -500


def test_start_on_triangle_edge():
    # Start via in line with two neighbouring vias (AD4 on B.Cu, GbaCartBreakout): the
    # triangulation has an edge between their grown tips that runs through the start
    # point. The sleeve must not begin in the triangle on the far side of that edge.
    obs = [(Point(-1.5 * MM, 0), 0.5 * MM), (Point(1.553 * MM, 0), 0.5 * MM)]
    path = mm([(0, 0), (-4.025, 4.025), (-8.205, 9.838), (-8.205, 10.155), (-8.9, 10.85)])
    r = pull_taut(path, obs)
    assert r == [path[0], path[-1]]
