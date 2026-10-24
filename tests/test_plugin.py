"""Plugin glue: chains, fixed points, obstacles and rules, against a mock board."""

import pytest
from kipy.board_types import FootprintInstance
from shapely.geometry import LineString, Point

import rubberband_action
from mock_board import MockBoard, _set_id, created_arcs, created_path, edge, pad, run, track, via


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
