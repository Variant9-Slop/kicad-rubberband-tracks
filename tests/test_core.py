"""Geometry core: shortest path with fixed topology, clearance never reduced."""

import math
import random

import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from rubberband_core import (
    RubberBandError, _circumcentre, _cross, fit_arcs, path_length, pull_taut, sample_elements,
)

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


# ---------------------------------------------------------------- arcs

def kicad_roundrect(x0, y0, x1, y1, rad, err=5000):
    """Rounded-rectangle pad as KiCad exports it: corners polygonised with chord error `err`."""
    n = max(2, int(math.ceil((math.pi / 2) / (2 * math.acos(1 - err / rad)))))
    pts = []
    for (cx, cy), a0 in (((x1 - rad, y1 - rad), 0), ((x0 + rad, y1 - rad), 90),
                         ((x0 + rad, y0 + rad), 180), ((x1 - rad, y0 + rad), 270)):
        for i in range(n + 1):
            a = math.radians(a0) + (math.pi / 2) * i / n
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return Polygon(pts)


def arcs_of(elements):
    return [e for e in elements if e[0] == "arc"]


def arc_centre_radius(e):
    c = _circumcentre(*e[1:])
    return c, math.dist(c, e[1])


def check_elements(elements, path, obstacles, tol=6.0):
    """Connected, same ends, and no closer to anything than required (or than `path`)."""
    assert elements[0][1] == path[0] and elements[-1][-1] == path[-1]
    for a, b in zip(elements, elements[1:]):
        assert a[-1] == b[1]
    curve = LineString(sample_elements(elements, 0.5))
    line = LineString(path)
    for g, d in obstacles:
        assert curve.distance(g) >= min(d, line.distance(g)) - tol


def test_arc_around_via():
    obs = [(Point(5 * MM, 0), 1 * MM)]
    path = pull_taut(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    el = fit_arcs(path, obs, margin=1000)
    assert [e[0] for e in el] == ["line", "arc", "line"]
    c, r = arc_centre_radius(el[1])
    assert math.dist(c, (5 * MM, 0)) < 1 and r == pytest.approx(1 * MM + 1000, abs=1)
    check_elements(el, path, obs)
    # analytic: tangents and arc around a circle of radius 1.001 mm
    R = 1.001
    t = math.sqrt(5 ** 2 - R ** 2)
    expected = 2 * t + 2 * R * math.asin(R / 5)
    assert path_length(sample_elements(el)) / MM == pytest.approx(expected, abs=1e-5)


def test_arc_around_square_pad_corner():
    pad = box(4 * MM, -1 * MM, 6 * MM, 1 * MM)
    obs = [(pad, 0.3 * MM)]
    path = pull_taut(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    el = fit_arcs(path, obs)
    arcs = arcs_of(el)
    assert len(arcs) == 2
    centres = sorted(arc_centre_radius(e)[0] for e in arcs)
    assert math.dist(centres[0], (4 * MM, 1 * MM)) < 1 and math.dist(centres[1], (6 * MM, 1 * MM)) < 1
    check_elements(el, path, obs)


def test_arc_around_rounded_rectangle_corner_is_one_arc():
    # wraps the top-left corner by ~90 degrees, touching several of its polygon points
    rad, d = 0.25 * MM, 0.3 * MM
    pad = kicad_roundrect(4 * MM, -1 * MM, 6 * MM, 1 * MM, rad)
    obs = [(pad, d)]
    path = pull_taut(mm([(3.6, -2), (3, 3), (8, 1.4)]), obs)
    el = fit_arcs(path, obs)
    assert [e[0] for e in el] == ["line", "arc", "line"]
    c, r = arc_centre_radius(el[1])
    assert math.dist(c, (4.25 * MM, 0.75 * MM)) < 1000
    assert r == pytest.approx(rad + d, abs=1000)
    check_elements(el, path, obs)


def test_short_wrap_of_rounded_corner_stays_smooth():
    # touches only two polygon points of each corner: no circle to fit, arcs around each
    pad = kicad_roundrect(4 * MM, -1 * MM, 6 * MM, 1 * MM, 0.25 * MM)
    obs = [(pad, 0.3 * MM)]
    path = pull_taut(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    el = fit_arcs(path, obs)
    assert all(e[0] == "line" for e in el[::2]) and all(e[0] == "arc" for e in el[1::2])
    check_elements(el, path, obs)


def test_s_bend_between_vias():
    obs = [(Point(3 * MM, 0), 0.8 * MM), (Point(7 * MM, 0), 0.8 * MM)]
    path = pull_taut(mm([(0, 0), (0, 2), (5, 2), (5, -2), (10, -2), (10, 0)]), obs)
    el = fit_arcs(path, obs, margin=1000)
    arcs = arcs_of(el)
    assert len(arcs) == 2
    assert _cross(*arcs[0][1:]) * _cross(*arcs[1][1:]) < 0      # opposite turns
    check_elements(el, path, obs)


def test_arcs_in_exact_width_gap():
    d = 0.175 * MM
    pads = [box(x * MM, 0, x * MM + 0.3 * MM, 1.0 * MM) for x in (0, 0.65)]
    mid = (0.3 + 0.65) / 2
    path = pull_taut(mm([(mid - 1, -2), (mid - 1, -1), (mid, -0.5), (mid, 1.5), (mid + 1, 2), (mid + 1, 3)]),
                     [(p, d) for p in pads])
    obs = [(p, d) for p in pads]
    for margin in (1000, 0):
        try:
            el = fit_arcs(path, obs, margin=margin)
            break
        except RubberBandError:
            continue
    check_elements(el, path, obs)


def test_arcs_random_obstacle_fields():
    rnd = random.Random(2)
    fitted = 0
    for _ in range(60):
        obs = []
        for _ in range(rnd.randint(1, 20)):
            x, y = rnd.uniform(0, 20) * MM, rnd.uniform(0, 20) * MM
            kind = rnd.random()
            if kind < .4:
                obs.append((Point(x, y), rnd.uniform(.2, 1.5) * MM))
            elif kind < .7:
                obs.append((box(x, y, x + rnd.uniform(.1, 3) * MM, y + rnd.uniform(.1, 3) * MM),
                            rnd.uniform(.05, .5) * MM))
            else:
                w, h = rnd.uniform(.5, 3) * MM, rnd.uniform(.5, 3) * MM
                obs.append((kicad_roundrect(x, y, x + w, y + h, min(w, h) * rnd.uniform(.1, .5)),
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
        path = [(round(x), round(y)) for x, y in pull_taut(pts, obs)]
        path[0], path[-1] = pts[0], pts[-1]
        for margin in (1000, 0):
            try:
                el = fit_arcs(path, obs, margin=margin)
            except RubberBandError:
                continue
            check_elements(el, path, obs)
            fitted += 1
            break
    assert fitted >= 40
