"""Minimal stand-in for kipy's Board, enough to run RubberBand.run() without KiCad."""

import json
import types
from pathlib import Path

from kipy.board_types import ArcTrack, BoardLayer, BoardSegment, Net, Pad, Track, Via
from kipy.geometry import PolygonWithHoles, PolyLine, PolyLineNode, Vector2
from kipy.proto.board import board_types_pb2
from kipy.project_types import NetClass
from shapely.geometry import box

import rubberband_action
from rubberband_core import sample_elements

MM = 1_000_000
F_CU = BoardLayer.BL_F_Cu
_next_id = [0]


def _set_id(item):
    _next_id[0] += 1
    item.proto.id.value = f"id{_next_id[0]}"
    return item


def _v(x, y):
    return Vector2.from_xy(int(round(x * MM)), int(round(y * MM)))


def net(name):
    n = Net()
    n.name = name
    return n


def track(a, b, net_name="A", width=0.25):
    t = _set_id(Track())
    t.start, t.end = _v(*a), _v(*b)
    t.width = int(width * MM)
    t.layer = F_CU
    t.net = net(net_name)
    return t


def pad(x, y, net_name, size=1.0, height=None):
    """Rectangular SMD pad (square unless height is given); its polygon is served by
    get_pad_shapes_as_polygons."""
    height = size if height is None else height
    p = _set_id(Pad())
    p.position = _v(x, y)
    p.net = net(net_name)
    outline = PolyLine()
    for cx, cy in box((x - size / 2) * MM, (y - height / 2) * MM,
                      (x + size / 2) * MM, (y + height / 2) * MM).exterior.coords[:-1]:
        outline.append(PolyLineNode.from_xy(int(cx), int(cy)))
    poly = PolygonWithHoles()
    poly.outline = outline
    p._mock_poly = poly
    # the pad stack says the same: a rectangle of that size (the plugin checks the two agree)
    cl = p.proto.pad_stack.copper_layers[0]
    cl.shape = board_types_pb2.PSS_RECTANGLE
    cl.size.x_nm, cl.size.y_nm = int(round(size * MM)), int(round(height * MM))
    return p


def via(x, y, net_name, diameter=0.6, drill=0.3):
    v = _set_id(Via())
    v.position = _v(x, y)
    v.net = net(net_name)
    v.diameter = int(diameter * MM)
    v.drill_diameter = int(drill * MM)
    return v


def edge(a, b):
    s = _set_id(BoardSegment())
    s.start, s.end = _v(*a), _v(*b)
    s.layer = BoardLayer.BL_Edge_Cuts
    return s


class MockBoard:
    def __init__(self, project_dir: Path, tracks=(), vias=(), pads=(), shapes=(), footprints=(),
                 selection=(), clearance=0.2, rules=None):
        self.tracks, self.vias, self.pads = list(tracks), list(vias), list(pads)
        self.shapes, self.footprints = list(shapes), list(footprints)
        self.selection = list(selection)
        self.clearance = int(clearance * MM)
        self.project_dir = project_dir
        self.created, self.removed = [], []
        rules = rules or {"min_copper_edge_clearance": 0.5, "min_hole_clearance": 0.25,
                          "min_clearance": 0.0}
        (project_dir / "mock.kicad_pro").write_text(
            json.dumps({"board": {"design_settings": {"rules": rules}}}))

    # --- reads
    def get_selection(self, types=None): return self.selection
    def get_tracks(self): return self.tracks
    def get_vias(self): return self.vias
    def get_pads(self): return self.pads
    def get_zones(self): return []
    def get_shapes(self): return self.shapes
    def get_footprints(self): return self.footprints
    def get_layer_name(self, layer): return "F.Cu"

    def get_project(self):
        return types.SimpleNamespace(path=str(self.project_dir), name="mock")

    def get_nets(self):
        names = {i.net.name for i in self.tracks + self.vias + self.pads}
        return [net(n) for n in sorted(names)]

    def get_netclass_for_nets(self, nets):
        nc = NetClass()
        nc.proto.name = "Default"
        nc.clearance = self.clearance
        return {n.name: nc for n in nets}

    def check_padstack_presence_on_layers(self, items, layer):
        return {i: {layer: True} for i in items}

    def get_pad_shapes_as_polygons(self, pads, layer):
        if isinstance(pads, list):
            return [p._mock_poly for p in pads]
        return pads._mock_poly

    # --- writes
    def create_items(self, items):
        items = list(items)
        self.created.extend(items)
        return items

    def remove_items(self, items):
        self.removed.extend(items)

    def clear_selection(self): pass
    def add_to_selection(self, items): pass


def run(board) -> rubberband_action.RubberBand:
    rb = rubberband_action.RubberBand.__new__(rubberband_action.RubberBand)
    rb.board = board
    rb.messages = []
    rb.run()
    return rb


def created_path(board, net_name=None):
    """Points of the created tracks (in mm), in creation order; arcs are followed to
    within 0.05 nm."""
    el = []
    for t in board.created:
        if net_name is None or t.net.name == net_name:
            xy = lambda v: (v.x, v.y)
            el.append(("arc", xy(t.start), xy(t.mid), xy(t.end)) if isinstance(t, ArcTrack)
                      else ("line", xy(t.start), xy(t.end)))
    return [(x / MM, y / MM) for x, y in sample_elements(el, 0.05)]


def created_arcs(board, net_name=None):
    return [t for t in board.created if isinstance(t, ArcTrack) and (net_name is None or t.net.name == net_name)]
