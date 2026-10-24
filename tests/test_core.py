"""Geometry core: shortest path with fixed topology, clearance never reduced."""

import math
import random

import pytest
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

import shapely

from rubberband_core import (
    CHECK_TOL, Ob, RubberBandError, _circumcentre, _cross, arc_band, exact_path, path_length,
    pull_taut, sample_elements,
)
from rubberband_core import min_margin as exact_margin

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


# ---------------------------------------------------------------- exact path

def roundrect(x0, y0, x1, y1, rad):
    """A rounded-rectangle pad as the plugin builds it: corner centres + corner radius."""
    return box(x0 + rad, y0 + rad, x1 - rad, y1 - rad), rad


def arcs_of(elements):
    return [e for e in elements if e[0] == "arc"]


def arc_centre_radius(e):
    c = _circumcentre(*e[1:])
    return c, math.dist(c, e[1])


def check_elements(elements, path, obstacles, tol=3.0):
    """Connected, same ends, clear of every true shape by the required distance."""
    assert elements[0][1] == path[0] and elements[-1][-1] == path[-1]
    for a, b in zip(elements, elements[1:]):
        assert math.dist(a[-1], b[1]) < 1e-6
    assert exact_margin(elements, obstacles) >= -tol


def solve(path, obs):
    return exact_path(pull_taut(path, obs), obs)


def test_arc_around_via():
    obs = [Ob(Point(5 * MM, 0), 1 * MM)]
    el = solve(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    assert [e[0] for e in el] == ["line", "arc", "line"]
    c, r = arc_centre_radius(el[1])
    assert math.dist(c, (5 * MM, 0)) < 1 and r == pytest.approx(1 * MM, abs=3)
    check_elements(el, mm([(0, 0), (10, 0)]), obs)
    # analytic: tangents and arc round a 1 mm circle
    t = math.sqrt(5 ** 2 - 1)
    assert path_length(sample_elements(el)) / MM == pytest.approx(2 * t + 2 * math.asin(1 / 5), abs=1e-5)


def test_arcs_around_square_pad_corners():
    obs = [Ob(box(4 * MM, -1 * MM, 6 * MM, 1 * MM), 0.3 * MM)]
    el = solve(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    assert [e[0] for e in el] == ["line", "arc", "line", "arc", "line"]
    centres = sorted(arc_centre_radius(e)[0] for e in arcs_of(el))
    assert math.dist(centres[0], (4 * MM, 1 * MM)) < 1 and math.dist(centres[1], (6 * MM, 1 * MM)) < 1
    check_elements(el, mm([(0, 0), (10, 0)]), obs)


@pytest.mark.parametrize("path", [[(3.6, -2), (3, 3), (8, 1.4)], [(0, 0), (0, 3), (10, 3), (10, 0)]])
def test_rounded_rectangle_corner_is_one_arc(path):
    core, rad = roundrect(4 * MM, -1 * MM, 6 * MM, 1 * MM, 0.25 * MM)
    obs = [Ob(core, rad + 0.3 * MM)]
    el = solve(mm(path), obs)
    for e in arcs_of(el):
        c, r = arc_centre_radius(e)
        assert min(math.dist(c, (x * MM, y * MM)) for x in (4.25, 5.75) for y in (-0.75, 0.75)) < 1
        assert r == pytest.approx(0.55 * MM, abs=3)
    check_elements(el, mm([path[0], path[-1]]), obs)


def test_s_bend_between_vias():
    obs = [Ob(Point(3 * MM, 0), 0.8 * MM), Ob(Point(7 * MM, 0), 0.8 * MM)]
    el = solve(mm([(0, 0), (0, 2), (5, 2), (5, -2), (10, -2), (10, 0)]), obs)
    arcs = arcs_of(el)
    assert len(arcs) == 2 and _cross(*arcs[0][1:]) * _cross(*arcs[1][1:]) < 0
    check_elements(el, mm([(0, 0), (10, 0)]), obs)


def test_u_turn_round_via():
    obs = [Ob(Point(5 * MM, 0), 1 * MM)]
    el = solve(mm([(4, -3), (4, -1.5), (3.5, 0), (5, 1.6), (6.5, 0), (6, -1.5), (6, -3)]), obs)
    assert all(arc_centre_radius(e)[1] == pytest.approx(1 * MM, abs=3) for e in arcs_of(el))
    check_elements(el, mm([(4, -3), (6, -3)]), obs)


def test_exact_width_gap():
    d = 0.175 * MM
    pads = [box(x * MM, 0, x * MM + 0.3 * MM, 1.0 * MM) for x in (0, 0.65)]
    mid = (0.3 + 0.65) / 2
    path = mm([(mid - 1, -2), (mid - 1, -1), (mid, -0.5), (mid, 1.5), (mid + 1, 2), (mid + 1, 3)])
    line = LineString(path)
    obs = [Ob(p, d, line.distance(p)) for p in pads]
    el = solve(path, obs)
    check_elements(el, [path[0], path[-1]], [Ob(p, min(d, line.distance(p))) for p in pads])


def test_arc_obstacle_is_wrapped_round_its_centre():
    # another track's arc, given as a band plus its circle: one concentric arc
    arc = (5 * MM, 0.0, 0.6 * MM, math.radians(130), math.radians(-80))
    obs = [Ob(arc_band(arc, 1000.0), 0.45 * MM, None, arc)]
    el = solve(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    assert [e[0] for e in el] == ["line", "arc", "line"]
    c, r = arc_centre_radius(el[1])
    assert math.dist(c, (5 * MM, 0)) < 1 and r == pytest.approx(1.05 * MM, abs=3)
    check_elements(el, mm([(0, 0), (10, 0)]), obs)


def test_short_arc_obstacle_wraps_its_ends_too():
    # a short arc: the path's tangents meet it beyond its ends, so it wraps the end caps
    # as well, each at the clearance, all joined smoothly
    arc = (5 * MM, 0.0, 0.6 * MM, math.radians(97), math.radians(-14))
    obs = [Ob(arc_band(arc, 1000.0), 0.45 * MM, None, arc)]
    el = solve(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), obs)
    assert [e[0] for e in el] == ["line", "arc", "arc", "arc", "line"]
    check_elements(el, mm([(0, 0), (10, 0)]), obs)


def test_original_closer_than_rules_is_kept_not_tightened():
    # original 0.25 mm from a via that wants 0.3: the result may be 0.25, no closer
    obs = [Ob(Point(5 * MM, 0), 0.3 * MM, 0.25 * MM)]
    el = solve(mm([(0, 0), (0, 0.6), (10, 0.6), (10, 0)]), obs)
    assert exact_margin(el, [Ob(Point(5 * MM, 0), 0.25 * MM)]) >= -3


def test_original_a_hair_too_close_is_put_right():
    # 0.7 um short of the rule (as earlier versions left tracks): pushed out to the rule
    obs = [Ob(Point(5 * MM, 0), 0.3 * MM, 0.3 * MM - 700)]
    el = solve(mm([(0, 0), (0, 0.6), (10, 0.6), (10, 0)]), obs)
    assert exact_margin(el, obs) >= 0


def test_exact_random_obstacle_fields():
    rnd = random.Random(2)
    done = 0
    for _ in range(80):
        obs = []
        for _ in range(rnd.randint(1, 20)):
            x, y = rnd.uniform(0, 20) * MM, rnd.uniform(0, 20) * MM
            kind = rnd.random()
            if kind < .4:
                obs.append(Ob(Point(x, y), rnd.uniform(.2, 1.5) * MM))
            elif kind < .7:
                obs.append(Ob(box(x, y, x + rnd.uniform(.1, 3) * MM, y + rnd.uniform(.1, 3) * MM),
                              rnd.uniform(.05, .5) * MM))
            else:
                w, h = rnd.uniform(.5, 3) * MM, rnd.uniform(.5, 3) * MM
                core, rad = roundrect(x, y, x + w, y + h, min(w, h) * rnd.uniform(.1, .5))
                obs.append(Ob(core, rad + rnd.uniform(.05, .5) * MM))
        blocked = unary_union([o.geom.buffer(o.d) for o in obs])
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
        el = solve(pts, obs)
        check_elements(el, [pts[0], pts[-1]], obs)
        # every arc is centred on a feature: a point, a corner, or a corner centre
        feats = [c for o in obs for c in shapely.get_coordinates(o.geom)]
        for e in arcs_of(el):
            c, _ = arc_centre_radius(e)
            assert min(math.dist(c, f) for f in feats) < 1
        assert path_length(sample_elements(el)) <= path_length(pts) + 1
        done += 1
    assert done >= 40


def test_reference_keeps_full_clearance_despite_creep():
    # A neighbour the path has crept 3 um inside of over repeated passes: with the
    # original gap as reference the result is put back at the full clearance (without it,
    # 3 um closer than the rule would pass as "the original was already closer").
    via = Point(5 * MM, 0)
    taut = pull_taut(mm([(0, 0), (0, 3), (10, 3), (10, 0)]), [Ob(via, 1 * MM - 3000)])
    el = exact_path(taut, [Ob(via, 1 * MM, 1 * MM)])
    assert exact_margin(el, [Ob(via, 1 * MM)]) >= 0
    crept = exact_path(taut, [Ob(via, 1 * MM)])           # default: the current distance
    assert exact_margin(crept, [Ob(via, 1 * MM)]) < -2000


# Squeezes: a gap a few nm narrower than both sides want (neighbours are each allowed to
# end up CHECK_TOL closer than asked). The shortfall is shared, so both sides stay within it.
SQUEEZE = 7.0


def test_squeeze_between_wrap_and_straight_side():
    # over a via, under a straight track: the corridor is SQUEEZE short of 2 mm
    d, y = 1 * MM, 1 * MM - SQUEEZE / 2
    via, track = Point(5 * MM, 0), LineString([(3 * MM, 2 * MM - SQUEEZE), (7 * MM, 2 * MM - SQUEEZE)])
    obs = [Ob(via, d, y), Ob(track, d, y)]
    path = [(0, 0), (4 * MM, y), (6 * MM, y), (10 * MM, 0)]
    el = solve(path, obs)
    check_elements(el, [path[0], path[-1]], obs, tol=CHECK_TOL)


def test_squeeze_between_opposite_wraps():
    # under one via, over the other: their clearance circles overlap by SQUEEZE
    d, L = 1 * MM, 2 * MM - SQUEEZE
    m = (5 * MM, 5 * MM)
    a, b = Point(m[0], m[1] - L / 2), Point(m[0], m[1] + L / 2)
    obs = [Ob(a, d, L / 2), Ob(b, d, L / 2)]
    path = [(m[0] + x * MM, m[1] + y * MM) for x, y in [(-3, 1.5), (-1, 0), (0, 0), (1, 0), (3, -1.5)]]
    el = solve(path, obs)
    check_elements(el, [path[0], path[-1]], obs, tol=CHECK_TOL)
