"""Plugin glue: chains, fixed points, obstacles and rules, against a mock board."""

import math

import pytest
from kipy.board_types import FootprintInstance
from kipy.geometry import arc_center
from shapely.geometry import LineString, Point

import rubberband_action
from mock_board import MM, MockBoard, _set_id, created_arcs, created_path, edge, pad, run, track, via


def around_via_board(tmp_path, selection_slice=slice(None), **kw):
    segs = [track((0, 0), (0, 3)), track((0, 3), (10, 3)), track((10, 3), (10, 0))]
    b = MockBoard(tmp_path, tracks=segs, vias=[via(5, 0, "B")],
                  pads=[pad(0, 0, "A"), pad(10, 0, "A")], selection=segs[selection_slice], **kw)
    return b, segs


def test_full_trace_hugs_via(tmp_path):
    b, _ = around_via_board(tmp_path)
    rb = run(b)
    assert rb.messages == []
    pts = created_path(b)
    assert pts[0] == (0, 0) and pts[-1] == (10, 0)
    # via r 0.3 + clearance 0.2 + half width 0.125
    assert LineString(pts).distance(Point(5, 0)) >= 0.625 - 1e-6
    assert len(b.removed) == 3


def test_already_straight_segment(tmp_path):
    b, _ = around_via_board(tmp_path, slice(1, 2))
    assert run(b).messages == ["Nothing to change: selection is already taut."]
    assert b.created == []


def test_partial_selection_keeps_junction(tmp_path):
    b, _ = around_via_board(tmp_path, slice(0, 2))
    run(b)
    assert created_path(b) == [(0, 0), (10, 3)]


def test_branch_point_stays_fixed(tmp_path):
    segs = [track((0, 0), (0, 3)), track((0, 3), (10, 3)), track((10, 3), (10, 0))]
    branch = track((0, 3), (-5, 3))
    b = MockBoard(tmp_path, tracks=segs + [branch], vias=[via(5, 0, "B")],
                  pads=[pad(0, 0, "A"), pad(10, 0, "A")], selection=segs)
    run(b)
    assert created_path(b)[0] == (0, 3)


def test_empty_selection(tmp_path):
    b = MockBoard(tmp_path)
    assert "Select one or more" in run(b).messages[0]


def test_footprint_edge_cuts_on_concave_board(tmp_path):
    outline = [(0, 0), (5, 0), (5, 5), (10, 5), (10, 10), (0, 10), (0, 0)]
    fp = FootprintInstance()
    fp.definition.items = [edge(a, b) for a, b in zip(outline, outline[1:])]
    segs = [track((2, 1), (2, 8)), track((2, 8), (9, 8)), track((9, 8), (9, 6))]
    b = MockBoard(tmp_path, tracks=segs, pads=[pad(2, 1, "A", 0.5), pad(9, 6, "A", 0.5)],
                  footprints=[fp], selection=segs)
    assert run(b).messages == []
    # edge clearance 0.5 + half width 0.125 around the inner corner
    assert LineString(created_path(b)).distance(Point(5, 5)) >= 0.625 - 1e-6


def test_hole_clearance_from_project_file(tmp_path):
    segs = [track((0, 0), (0, 3)), track((0, 3), (10, 3)), track((10, 3), (10, 0))]
    b = MockBoard(tmp_path, tracks=segs, vias=[via(5, 0, "B", 0.45, 0.3)],
                  pads=[pad(0, 0, "A"), pad(10, 0, "A")], selection=segs,
                  rules={"min_copper_edge_clearance": 0.5, "min_hole_clearance": 0.35})
    run(b)
    # copper: 0.225 + 0.2 + 0.125 = 0.55; hole: 0.15 + 0.35 + 0.125 = 0.625
    assert LineString(created_path(b)).distance(Point(5, 0)) >= 0.625 - 1e-6


def test_nested_nets_settle_regardless_of_order(tmp_path):
    a = [track((0, 0), (0, 4)), track((0, 4), (10, 4)), track((10, 4), (10, 0))]
    bb = [track((-1, 0), (-1, 5), "B"), track((-1, 5), (11, 5), "B"), track((11, 5), (11, 0), "B")]
    pads = [pad(0, 0, "A", 0.4), pad(10, 0, "A", 0.4), pad(-1, 0, "B", 0.4), pad(11, 0, "B", 0.4)]
    b = MockBoard(tmp_path, tracks=a + bb, pads=pads, selection=bb + a)   # outer net first
    assert run(b).messages == []
    assert created_path(b, "A") == [(0, 0), (10, 0)]
    path_b = LineString(created_path(b, "B"))
    assert max(y for _, y in path_b.coords) < 0.6       # B pulled down onto A
    assert path_b.distance(LineString([(0, 0), (10, 0)])) >= 0.45 - 1e-6


def test_bend_on_edge_of_end_pad_is_not_fixed(tmp_path):
    # Last bend sits exactly on the edge of the pad the run ends in (C11 -> SW2 on
    # GbaCartBreakout): it must not split off the final segment.
    segs = [track((0, 0), (1.1, 1.1)), track((1.1, 1.1), (6.4625, 1.1)),
            track((6.4625, 1.1), (6.8625, 0.7))]
    b = MockBoard(tmp_path, tracks=segs, selection=segs,
                  pads=[pad(0, 0, "A", 1.075, 0.95), pad(6.8625, 0.7, "A", 2, 0.8)])
    assert run(b).messages == []
    assert created_path(b) == [(0, 0), (6.8625, 0.7)]
    assert len(b.removed) == 3


def test_bend_inside_pad_mid_run_stays_fixed(tmp_path):
    # A run that passes through another pad of its net keeps touching it.
    segs = [track((0, 0), (5, 3)), track((5, 3), (10, 0))]
    b = MockBoard(tmp_path, tracks=segs, selection=segs,
                  pads=[pad(0, 0, "A"), pad(5, 3, "A"), pad(10, 0, "A")])
    assert run(b).messages == ["Nothing to change: selection is already taut."]


def _ends_joined(items):
    for a, b in zip(items, items[1:]):
        assert (a.end.x, a.end.y) == (b.start.x, b.start.y)


def test_via_wrap_becomes_one_arc(tmp_path):
    b, _ = around_via_board(tmp_path)
    run(b)
    assert [type(t).__name__ for t in b.created] == ["Track", "ArcTrack", "Track"]
    _ends_joined(b.created)
    # 0.3 + 0.2 + 0.125 required, plus ARC_MARGIN
    assert LineString(created_path(b)).distance(Point(5, 0)) == pytest.approx(0.626, abs=5e-6)


def test_arcs_can_be_turned_off(tmp_path, monkeypatch):
    monkeypatch.setattr(rubberband_action, "USE_ARCS", False)
    b, _ = around_via_board(tmp_path)
    run(b)
    assert len(b.created) > 3 and not created_arcs(b)
    _ends_joined(b.created)


def test_nested_nets_both_get_arcs(tmp_path):
    a = [track((0, 0), (0, 4)), track((0, 4), (10, 4)), track((10, 4), (10, 0))]
    bb = [track((-1, 0), (-1, 5), "B"), track((-1, 5), (11, 5), "B"), track((11, 5), (11, 0), "B")]
    pads = [pad(0, 0, "A", 0.4), pad(10, 0, "A", 0.4), pad(-1, 0, "B", 0.4), pad(11, 0, "B", 0.4)]
    b = MockBoard(tmp_path, tracks=a + bb, pads=pads, vias=[via(5, 1, "C")], selection=bb + a)
    assert run(b).messages == []
    assert created_arcs(b, "A") and created_arcs(b, "B")
    pa, pb = LineString(created_path(b, "A")), LineString(created_path(b, "B"))
    assert pa.distance(Point(5, 1)) >= 0.625 - 1e-6
    assert pb.distance(pa) >= 0.45 - 1e-6


def test_rerun_on_arcs_changes_nothing(tmp_path):
    b, _ = around_via_board(tmp_path)
    run(b)
    for t in b.created:
        _set_id(t)
    again = MockBoard(tmp_path, tracks=b.created, vias=b.vias, pads=b.pads, selection=b.created)
    assert run(again).messages == ["Nothing to change: selection is already taut."]


def test_nested_bends_rounded_one_by_one(tmp_path):
    # Tracks at 0.45 mm pitch turning right around a sharp inner corner; round them one at
    # a time from the inside out. Each wraps the arc made just before (read back from the
    # board as an ArcTrack) and must come out as a single arc, not slivers at its joints.
    inner = [track((1.0, 0), (1.0, 5.0), "N0"), track((1.0, 5.0), (10, 5.0), "N0")]
    pads = [pad(1.0, 0, "N0", .3), pad(10, 5.0, "N0", .3)]
    board_tracks = list(inner)
    for n in range(1, 4):
        x, y, name = 1.0 - 0.45 * n, 5.0 + 0.9 * n, f"N{n}"
        segs = [track((x, 0), (x, y), name), track((x, y), (10, y), name)]
        pads += [pad(x, 0, name, .3), pad(10, y, name, .3)]
        b = MockBoard(tmp_path, tracks=board_tracks + segs, pads=pads, selection=segs)
        assert run(b).messages == []
        assert [type(t).__name__ for t in b.created] == ["Track", "ArcTrack", "Track"]
        _ends_joined(b.created)
        prev = LineString(created_path(b))
        for t in b.created:
            _set_id(t)
        if n > 1:
            assert prev.distance(prev_path) >= 0.45 - 1e-6
        prev_path = prev
        board_tracks += b.created


def test_nested_bends_rounded_together(tmp_path, monkeypatch):
    # Same, all selected at once: the middle runs are squeezed between neighbours at
    # their final position and still get one arc each, never closer to the neighbours
    # than plain segments end up.
    def nested(use_arcs):
        monkeypatch.setattr(rubberband_action, "USE_ARCS", use_arcs)
        inner = [track((1.0, 0), (1.0, 5.0), "N0"), track((1.0, 5.0), (10, 5.0), "N0")]
        pads, sel = [pad(1.0, 0, "N0", .3), pad(10, 5.0, "N0", .3)], []
        for n in range(1, 4):
            x, y, name = 1.0 - 0.45 * n, 5.0 + 0.9 * n, f"N{n}"
            sel += [track((x, 0), (x, y), name), track((x, y), (10, y), name)]
            pads += [pad(x, 0, name, .3), pad(10, y, name, .3)]
        b = MockBoard(tmp_path, tracks=inner + sel, pads=pads, selection=sel)
        assert run(b).messages == []
        paths = [LineString([(1, 0), (1, 5), (10, 5)])] + \
            [LineString(created_path(b, f"N{n}")) for n in range(1, 4)]
        return b, [a.distance(c) for a, c in zip(paths, paths[1:])]

    _, gaps_segments = nested(False)
    b, gaps = nested(True)
    for n in range(1, 4):
        assert [type(t).__name__ for t in b.created if t.net.name == f"N{n}"] == ["Track", "ArcTrack", "Track"]
    for g, gs in zip(gaps, gaps_segments):
        assert g >= gs - 5e-6


def test_tracks_around_via_rounded_one_by_one(tmp_path):
    # Round a track around a via, then each next track out (at minimum pitch) around the
    # previous one: the arcs read back from the board give one concentric arc per track,
    # not one per polyline point, and the creep outwards doesn't hit the next original.
    def U(off, name):
        return [track((-off, -off), (-off, 3 + off), name), track((-off, 3 + off), (10 + off, 3 + off), name),
                track((10 + off, 3 + off), (10 + off, -off), name)]
    vias = [via(5, 0, "V")]
    runs = [U(0.45 * n, f"N{n}") for n in range(4)]
    pads = [pad(x, -0.45 * n, f"N{n}", .1) for n in range(4) for x in (-0.45 * n, 10 + 0.45 * n)]
    board = []
    for n, r in enumerate(runs):
        later = [t for rr in runs[n + 1:] for t in rr]
        b = MockBoard(tmp_path, tracks=board + r + later, vias=vias, pads=pads, selection=r)
        assert run(b).messages == []
        _ends_joined(b.created)
        assert all(math.hypot(t.end.x - t.start.x, t.end.y - t.start.y) > 10_000 for t in b.created)
        # one arc around the via; from the second track on, also one around each end of
        # the track inside it (its pads put those exactly at the clearance)
        centres = [arc_center(t.start, t.mid, t.end) for t in created_arcs(b)]
        assert sum(math.hypot(c.x - 5 * MM, c.y) < 1000 for c in centres) == 1
        assert len(centres) == (1 if n == 0 else 3)
        path = LineString(created_path(b))
        for t in later:
            assert path.distance(LineString([(t.start.x / MM, t.start.y / MM), (t.end.x / MM, t.end.y / MM)])) \
                >= 0.45 - 3e-6
        if n:
            # the originals sit at exactly the clearance: 2 nm (pull_taut's eps) + rounding
            assert path.distance(prev) >= 0.45 - 3e-6
        prev = path
        for t in b.created:
            _set_id(t)
        board += b.created


@pytest.mark.parametrize("deg, shape", [(0.2, ["Track", "Track"]), (1, ["Track", "ArcTrack", "Track"]),
                                        (5, ["Track", "ArcTrack", "Track"])])
def test_slight_bend_around_track_corner(tmp_path, deg, shape):
    # A neighbour pulled around a track that bends by a fraction of a degree to a few:
    # one arc, or one corner where an arc would be too flat to survive rounding to nm;
    # never a sliver of segment in between.
    t = math.radians(deg)
    corner = [track((0, 0), (5, 0), "O"), track((5, 0), (5 + 5 * math.cos(t), 5 * math.sin(t)), "O")]
    y = -0.45
    end = (5 + 5 * math.cos(t) - y * math.sin(t), y * math.cos(t) + 5 * math.sin(t))
    sel = [track((0, y), (4, y - 1), "A"), track((4, y - 1), (6, y - 1), "A"), track((6, y - 1), end, "A")]
    b = MockBoard(tmp_path, tracks=corner + sel, selection=sel,
                  pads=[pad(0, y, "A", .1), pad(sel[-1].end.x / MM, sel[-1].end.y / MM, "A", .1)])
    assert run(b).messages == []
    assert [type(t).__name__ for t in b.created] == shape
    _ends_joined(b.created)
    assert LineString(created_path(b)).distance(LineString([(0, 0), (5, 0)])) >= 0.45 - 3e-6
