"""KiCad IPC action: pull the selected track segments taut around other copper.

Select one or more track segments (or a whole trace), click the toolbar button.
Each run of selected, connected segments on the same layer/net/width is replaced
by the shortest path that passes every obstacle on the same side as before.
Points where the selection connects to anything else (pads, vias, unselected
tracks, branches) stay where they are.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict, deque

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from kipy import KiCad
from kipy.board_types import (
    ArcTrack, BoardArc, BoardBezier, BoardCircle, BoardLayer, BoardPolygon,
    BoardRectangle, BoardSegment, Track,
)
from kipy.geometry import Vector2, arc_center
from kipy.proto.board import board_types_pb2
from kipy.proto.common.types import KiCadObjectType

from rubberband_core import (
    Ob, RubberBandError, arc_band, arc_points, element_arc, exact_path, min_margin, path_length,
    pull_taut, sample_elements,
)

# ------------------------------------------------------------------ settings (nm)
EXTRA_MARGIN = 0               # added on top of every clearance (use for custom DRC rules)
DEFAULT_CLEARANCE = 200_000    # used when a net class has no clearance set
EDGE_CLEARANCE = None          # None = board setting "copper to edge clearance" (.kicad_pro)
HOLE_CLEARANCE = None          # None = board setting "copper to hole clearance" (.kicad_pro)
MAX_REDOS = 50                 # runs are redone when a neighbour moves; cap per run, on average
API_TIMEOUT_MS = 120_000       # per request; large boards need more than the 2 s default
AVOID_COPPER_ZONES = False     # False: zones of other nets are ignored (they refill around you)
ARC_MAX_ERROR = 1_000          # chord error when turning arcs into segments
PAD_POLY_ERROR = 5_000         # KiCad's own polygon approximation error for round pads
QUAD_SEGS = 32                 # circle resolution for Edge.Cuts circles
USE_ARCS = True                # exact lines and arcs; False: the plain segments pull_taut finds
MIN_SEGMENT = 10_000           # a result with a shorter straight piece is a sliver: run left as is
VIOLATION_TOL = 5              # an original closer than required by more than this may be fixed
BIG_GAIN = 50_000              # a result in more pieces is taken only if it is this much shorter


def _xy(v: Vector2) -> tuple[int, int]:
    return (int(v.x), int(v.y))


def _curve_circle(start: Vector2, mid: Vector2, end: Vector2):
    """(cx, cy, r, start angle, signed sweep) of the arc through start, mid and end, or
    None if it is straight."""
    c = arc_center(start, mid, end)
    if c is None:
        return None
    cx, cy = c.x, c.y
    a0 = math.atan2(start.y - cy, start.x - cx)
    am = math.atan2(mid.y - cy, mid.x - cx)
    a1 = math.atan2(end.y - cy, end.x - cx)

    def norm(a):
        return a % (2 * math.pi)
    sweep = norm(a1 - a0)
    if norm(am - a0) > sweep:          # mid not on the CCW sweep -> go the other way
        sweep -= 2 * math.pi
    return cx, cy, math.hypot(start.x - cx, start.y - cy), a0, sweep


def _shape_circle(s):
    if isinstance(s, BoardArc):
        return _curve_circle(s.start, s.mid, s.end)
    if isinstance(s, BoardCircle):
        return (s.center.x, s.center.y, s.radius(), 0.0, 2 * math.pi)
    return None


def _arc_points(start: Vector2, mid: Vector2, end: Vector2) -> list[tuple[float, float]]:
    """Arc as a polyline, chords within ARC_MAX_ERROR inside it."""
    circle = _curve_circle(start, mid, end)
    if circle is None:
        return [_xy(start), _xy(end)]
    return [_xy(start)] + arc_points(circle, ARC_MAX_ERROR)[1:-1] + [_xy(end)]


def _arc_band(start: Vector2, mid: Vector2, end: Vector2):
    """Arc as an obstacle (see rubberband_core.arc_band)."""
    circle = _curve_circle(start, mid, end)
    return LineString([_xy(start), _xy(end)]) if circle is None else arc_band(circle, ARC_MAX_ERROR)


def _item_elements(items, start) -> list[tuple]:
    """Tracks of a run, in order from `start`, as ("line", a, b) / ("arc", a, mid, b)."""
    out, cur = [], start
    for t in items:
        s, e = _xy(t.start), _xy(t.end)
        a, b = (s, e) if s == cur else (e, s)
        out.append(("arc", a, _xy(t.mid), b) if isinstance(t, ArcTrack) else ("line", a, b))
        cur = b
    return out


def _pad_shape(pad, layer, poly):
    """(geometry, copper half-size) of a pad on `layer`: its true shape for circles, ovals,
    rectangles and rounded rectangles, used only if it matches KiCad's polygon of the pad
    (so a mistake in reading rotation or offset falls back); otherwise that polygon."""
    fallback = (poly, PAD_POLY_ERROR)
    if poly is None:
        return fallback
    try:
        ps = pad.padstack
        cl = ps.copper_layer(layer)
        if cl is None and ps.copper_layers:
            if ps.type == board_types_pb2.PST_NORMAL:
                cl = ps.copper_layers[0]
            elif ps.type == board_types_pb2.PST_FRONT_INNER_BACK:
                outer = {BoardLayer.BL_F_Cu, BoardLayer.BL_B_Cu}
                cl = next((c for c in ps.copper_layers if c.layer not in outer), None)
        if cl is None:
            return fallback
        sx, sy, shape = cl.size.x, cl.size.y, cl.shape
        if sx <= 0 or sy <= 0:
            return fallback
        for sign in (1, -1):        # KiCad's y points down; try both senses of the angle
            a = sign * math.radians(ps.angle.degrees)
            ca, sa = math.cos(a), math.sin(a)
            ox, oy = cl.offset.x, cl.offset.y
            px, py = pad.position.x + ox * ca - oy * sa, pad.position.y + ox * sa + oy * ca
            at = lambda x, y: (px + x * ca - y * sa, py + x * sa + y * ca)

            def core(hx, hy):       # the shape's skeleton: point, segment or rectangle
                if hx <= 0 and hy <= 0:
                    return Point(at(0, 0))
                if hx <= 0 or hy <= 0:
                    return LineString([at(-hx, 0), at(hx, 0)]) if hx > 0 else LineString([at(0, -hy), at(0, hy)])
                return Polygon([at(-hx, -hy), at(hx, -hy), at(hx, hy), at(-hx, hy)])
            if shape == board_types_pb2.PSS_CIRCLE:
                geom, rho = core(0, 0), sx / 2
            elif shape == board_types_pb2.PSS_OVAL:
                rho = min(sx, sy) / 2
                geom = core(sx / 2 - rho, sy / 2 - rho)
            elif shape == board_types_pb2.PSS_RECTANGLE:
                geom, rho = core(sx / 2, sy / 2), 0.0
            elif shape == board_types_pb2.PSS_ROUNDRECT:
                rho = cl.corner_rounding_ratio * min(sx, sy)
                geom = core(sx / 2 - rho, sy / 2 - rho)
            else:
                return fallback
            outline = geom.buffer(rho, quad_segs=64) if rho > 0 else geom
            if outline.hausdorff_distance(poly) <= 2 * PAD_POLY_ERROR + 1000:
                return geom, rho
    except Exception:
        pass
    return fallback


def _track_points(t) -> list:
    if isinstance(t, ArcTrack):
        return _arc_points(t.start, t.mid, t.end)
    return [_xy(t.start), _xy(t.end)]


def _track_geom(t):
    """A track as an obstacle."""
    if isinstance(t, ArcTrack):
        return _arc_band(t.start, t.mid, t.end)
    return LineString(_track_points(t))


def _poly_from_proto(pwh) -> Polygon | None:
    def ring(polyline):
        pts = []
        for node in polyline.nodes:
            if node.has_point:
                pts.append(_xy(node.point))
            elif node.has_arc:
                a = node.arc
                pts.extend(_arc_points(a.start, a.mid, a.end))
        return pts
    outer = ring(pwh.outline)
    if len(outer) < 3:
        return None
    holes = [h for h in (ring(x) for x in pwh.holes) if len(h) >= 3]
    p = Polygon(outer, holes)
    return p if p.is_valid else p.buffer(0)


def _shape_outline(s):
    """Edge.Cuts / copper graphic -> shapely geometry of its outline (line-like)."""
    if isinstance(s, BoardSegment):
        return LineString([_xy(s.start), _xy(s.end)])
    if isinstance(s, BoardArc):
        return _arc_band(s.start, s.mid, s.end)
    if isinstance(s, BoardCircle):
        # band between the inscribed and circumscribed polygons (see _arc_band)
        (cx, cy), n, r = _xy(s.center), 4 * QUAD_SEGS, s.radius()
        ring = lambda rr, off: [(cx + rr * math.cos(2 * math.pi * (i + off) / n),
                                 cy + rr * math.sin(2 * math.pi * (i + off) / n)) for i in range(n)]
        return Polygon(ring(r / math.cos(math.pi / n), 0.5), [ring(r, 0)])
    if isinstance(s, BoardRectangle):
        (x0, y0), (x1, y1) = _xy(s.top_left), _xy(s.bottom_right)
        return Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]).exterior
    if isinstance(s, BoardPolygon):
        geoms = [p for p in (_poly_from_proto(x) for x in s.polygons) if p is not None]
        return unary_union(geoms).boundary if geoms else None
    if isinstance(s, BoardBezier):
        p0, p1, p2, p3 = (_xy(v) for v in (s.start, s.control1, s.control2, s.end))
        pts = []
        for i in range(33):
            t = i / 32
            u = 1 - t
            pts.append(tuple(u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t**3 * d
                             for a, b, c, d in zip(p0, p1, p2, p3)))
        return LineString(pts)
    return None


class Obstacle:
    __slots__ = ("geom", "net", "radius", "kind", "clearance_override", "circle")

    def __init__(self, geom, net, radius=0.0, kind="copper", clearance_override=None, circle=None):
        self.geom = geom            # raw (unbuffered) geometry
        self.net = net              # net name, "" = no net
        self.radius = radius        # copper half-size already implied by the geometry
        self.kind = kind            # copper | edge | keepout
        self.clearance_override = clearance_override
        self.circle = circle        # (cx, cy, r) of the arc or circle geom approximates


class RubberBand:
    def __init__(self):
        self.kicad = KiCad(timeout_ms=API_TIMEOUT_MS)
        self.board = self.kicad.get_board()
        self.messages: list[str] = []

    # ---------------------------------------------------------- board data
    def load(self):
        b = self.board
        self.tracks = list(b.get_tracks())
        self.vias = list(b.get_vias())
        self.pads = list(b.get_pads())
        self.zones = list(b.get_zones())
        self.shapes = list(b.get_shapes())

        nets = list(b.get_nets())
        classes = b.get_netclass_for_nets(nets) if nets else {}
        default = next((c.clearance for c in classes.values()
                        if c.name == "Default" and c.clearance), None) or DEFAULT_CLEARANCE
        self.clearance = defaultdict(lambda: default)
        for name, nc in classes.items():
            self.clearance[name] = nc.clearance or default

        self.pad_polys: dict[int, list] = {}
        self.via_cache: dict[int, list] = {}
        self.layer_index: dict[int, tuple] = {}
        self._anchors = None
        self._pad_net_cache = {}

        rules = self._project_rules()
        mm = lambda v: int(round(float(v) * 1e6))
        self.edge_clearance = EDGE_CLEARANCE if EDGE_CLEARANCE is not None else \
            mm(rules.get("min_copper_edge_clearance", 0.5))
        self.hole_clearance = HOLE_CLEARANCE if HOLE_CLEARANCE is not None else \
            mm(rules.get("min_hole_clearance", 0.25))
        board_min = mm(rules.get("min_clearance", 0))
        for k in list(self.clearance.keys()):
            self.clearance[k] = max(self.clearance[k], board_min)
        self.clearance.default_factory = lambda: max(default, board_min)

        # Edge.Cuts drawn on the board and inside footprints (absolute coordinates)
        self.edges = [(g, _shape_circle(sh)) for sh in self.shapes
                      if sh.layer == BoardLayer.BL_Edge_Cuts and (g := _shape_outline(sh)) is not None]
        try:
            for fp in b.get_footprints():
                for sh in fp.definition.shapes:
                    if sh.layer == BoardLayer.BL_Edge_Cuts:
                        g = _shape_outline(sh)
                        if g is not None:
                            self.edges.append((g, _shape_circle(sh)))
        except Exception as e:
            self.messages.append(f"Could not read footprint Edge.Cuts: {e}")

    def _project_rules(self) -> dict:
        """board.design_settings.rules from the .kicad_pro file (values in mm)."""
        candidates = []
        try:
            p = self.board.get_project()
            candidates.append(os.path.join(p.path, p.name + ".kicad_pro"))
        except Exception:
            pass
        try:
            doc = self.board.document
            if doc.board_filename:
                base = os.path.splitext(doc.board_filename)[0] + ".kicad_pro"
                candidates.append(base)
                candidates.append(os.path.join(doc.project.path, os.path.basename(base)))
        except Exception:
            pass
        for path in candidates:
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)["board"]["design_settings"]["rules"]
            except Exception:
                continue
        self.messages.append("Could not read board rules from the .kicad_pro file; using KiCad defaults "
                             "(edge 0.5 mm, hole 0.25 mm).")
        return {}

    def _pad_polygons(self, layer) -> list[tuple]:
        """[(pad, polygon or None)] for pads on `layer`."""
        if layer in self.pad_polys:
            return self.pad_polys[layer]
        present = self.board.check_padstack_presence_on_layers(self.pads, layer) if self.pads else {}
        on_layer = [p for p in self.pads if present.get(p, {}).get(layer, False)]
        polys = []
        if on_layer:
            got = self.board.get_pad_shapes_as_polygons(on_layer, layer)
            if len(got) != len(on_layer):     # library drops missing entries; fall back to one by one
                got = [self.board.get_pad_shapes_as_polygons(p, layer) for p in on_layer]
            polys = [(p, _poly_from_proto(g) if g is not None else None) for p, g in zip(on_layer, got)]
        self.pad_polys[layer] = polys
        return polys

    def _vias_on(self, layer) -> list:
        if layer not in self.via_cache:
            present = self.board.check_padstack_presence_on_layers(self.vias, layer) if self.vias else {}
            self.via_cache[layer] = [v for v in self.vias if present.get(v, {}).get(layer, False)]
        return self.via_cache[layer]

    def obstacles_for_layer(self, layer, own_net: str, region, moving: set[str]) -> list[Obstacle]:
        """Static obstacles near `region` that `own_net` must avoid."""
        if layer not in self.layer_index:
            obs = self._build_layer(layer, moving)
            self.layer_index[layer] = (obs, STRtree([o.geom for o in obs]) if obs else None)
        obs, tree = self.layer_index[layer]
        if tree is None:
            return []
        out = []
        for i in tree.query(region):
            o = obs[int(i)]
            if o.kind in ("copper", "hole") and own_net != "" and o.net == own_net:
                continue
            out.append(o)
        return out

    def _build_layer(self, layer, moving: set[str]) -> list[Obstacle]:
        """All obstacles on `layer` for every net; `moving` = ids of all selected tracks."""
        obs: list[Obstacle] = []
        for t in self.tracks:
            if t.layer != layer or t.id.value in moving:
                continue
            circle = _curve_circle(t.start, t.mid, t.end) if isinstance(t, ArcTrack) else None
            obs.append(Obstacle(_track_geom(t), t.net.name, t.width / 2, circle=circle))
        for v in self._vias_on(layer):
            obs.append(Obstacle(Point(_xy(v.position)), v.net.name, v.diameter / 2))
        # drill holes: copper-to-hole clearance, on every layer the hole passes through
        for v in self.vias:
            if v.drill_diameter > 0:
                obs.append(Obstacle(Point(_xy(v.position)), v.net.name, v.drill_diameter / 2, kind="hole"))
        for p in self.pads:
            dd = p.padstack.drill.diameter
            if dd.x > 0 and dd.y > 0:
                obs.append(Obstacle(Point(_xy(p.position)), p.net.name, max(dd.x, dd.y) / 2, kind="hole"))
        for pad, poly in self._pad_polygons(layer):
            if poly is None:
                continue
            override = None
            try:
                if pad.proto.HasField("copper_clearance_override"):
                    override = pad.proto.copper_clearance_override.value_nm
            except Exception:
                pass
            geom, rho = _pad_shape(pad, layer, poly)
            obs.append(Obstacle(geom, pad.net.name, rho, clearance_override=override))
        for z in self.zones:
            if layer not in z.layers:
                continue
            if z.type == board_types_pb2.ZT_RULE_AREA:
                if z.proto.rule_area_settings.keepout_tracks:
                    poly = _poly_from_proto(z.outline)
                    if poly is not None:
                        obs.append(Obstacle(poly, "", kind="keepout"))
            elif z.type == board_types_pb2.ZT_COPPER and AVOID_COPPER_ZONES:
                net = z.net.name if z.net else ""
                for pwh in z.filled_polygons.get(layer, []):
                    poly = _poly_from_proto(pwh)
                    if poly is not None:
                        obs.append(Obstacle(poly, net))
        for g, circle in self.edges:
            obs.append(Obstacle(g, "", kind="edge", circle=circle))
        for s in self.shapes:
            if s.layer == layer:
                net = s.net.name if s.net else ""
                g = _shape_outline(s)
                if g is not None:
                    obs.append(Obstacle(g, net, s.attributes.stroke.width / 2, circle=_shape_circle(s)))
        return obs

    def required(self, obs: list[Obstacle], own_net: str, half_width: float) -> list[tuple]:
        """(geometry, required centreline distance) for each obstacle."""
        own_cl = self.clearance[own_net]
        out = []
        for o in obs:
            if o.kind == "keepout":
                d = half_width
            elif o.kind == "edge":
                d = self.edge_clearance + half_width
            elif o.kind == "hole":
                d = o.radius + self.hole_clearance + half_width
            else:
                cl = max(own_cl, self.clearance[o.net] if o.net else own_cl,
                         o.clearance_override or 0)
                d = o.radius + cl + half_width
            out.append((o.geom, d + EXTRA_MARGIN))
        return out

    # ---------------------------------------------------------- chains
    def _pads_by_net(self, layer) -> dict:
        cache = self.__dict__.setdefault("_pad_net_cache", {})
        if layer not in cache:
            idx = defaultdict(list)
            for pad, poly in self._pad_polygons(layer):
                if poly is not None:
                    idx[pad.net.name].append(poly)
            cache[layer] = idx
        return cache[layer]

    def _anchor_index(self, selected) -> dict:
        """(layer, net) -> points where unselected same-net tracks end, plus via positions."""
        if getattr(self, "_anchors", None) is None:
            sel_ids = {t.id.value for t in selected}
            idx = defaultdict(set)
            vias_by_net = defaultdict(list)
            for v in self.vias:
                vias_by_net[v.net.name].append(_xy(v.position))
            for t in self.tracks:
                if t.id.value in sel_ids:
                    continue
                key = (t.layer, t.net.name)
                idx[key].add(_xy(t.start))
                idx[key].add(_xy(t.end))
            for t in selected:
                key = (t.layer, t.net.name)
                idx[key].update(vias_by_net.get(t.net.name, ()))
            self._anchors = idx
        return self._anchors

    def chains(self, selected):
        """Split the selection into runs of connected segments with fixed ends."""
        groups = defaultdict(list)
        for t in selected:
            groups[(t.layer, t.net.name, t.width)].append(t)

        result = []
        for (layer, net, width), items in groups.items():
            nodes = defaultdict(list)
            for t in items:
                nodes[_xy(t.start)].append(t)
                nodes[_xy(t.end)].append(t)

            anchors = self._anchor_index(selected).get((layer, net), set())
            pad_polys = self._pads_by_net(layer).get(net, [])

            def fixed(n):
                return len(nodes[n]) != 2 or n in anchors

            def pads_at(n):
                p = Point(n)
                return {i for i, poly in enumerate(pad_polys) if poly.covers(p)}

            used = set()
            for n in list(nodes):
                if not fixed(n):
                    continue
                for t in nodes[n]:
                    if t.id.value in used:
                        continue
                    chain, cur, seg = [], n, t
                    pts, joints = [n], [n]
                    while True:
                        used.add(seg.id.value)
                        chain.append(seg)
                        sp = _track_points(seg)
                        if _xy(seg.start) != cur:
                            sp = sp[::-1]
                        pts.extend(sp[1:])
                        cur = _xy(seg.end) if _xy(seg.start) == cur else _xy(seg.start)
                        joints.append(cur)
                        if fixed(cur):
                            break
                        seg = next((s for s in nodes[cur] if s.id.value not in used), None)
                        if seg is None:
                            break
                    # A run that passes through another pad of its net must keep touching it,
                    # so split there. Bends inside the pads the run ends in (even just on
                    # their edge) are free: the run is connected to those pads anyway.
                    end_pads = pads_at(joints[0]) | pads_at(joints[-1])
                    start_seg, start_pt = 0, 0
                    for k in range(1, len(chain)):
                        if pads_at(joints[k]) - end_pads:
                            end_pt = pts.index(joints[k], start_pt + 1)
                            result.append((layer, net, width, chain[start_seg:k], pts[start_pt:end_pt + 1]))
                            start_seg, start_pt = k, end_pt
                    result.append((layer, net, width, chain[start_seg:], pts[start_pt:]))
            if len(used) < len(items):
                self.messages.append(f"{len(items) - len(used)} segment(s) form a closed loop; skipped")
        return result

    # ---------------------------------------------------------- run
    def _dynamic(self, ch, exact=False) -> list[Ob]:
        """The other selected runs next to `ch`, as obstacles where they are now. For
        pull_taut they may be as close as they are now (its input has to stay outside its
        grown obstacles); for the exact path only as close as the two original routes were,
        or else repeated redos could creep closer and closer."""
        out = []
        for other in ch["neighbours"]:
            cl = max(self.clearance[ch["net"]], self.clearance[other["net"]])
            d = other["width"] / 2 + cl + ch["width"] / 2 + EXTRA_MARGIN
            ref = ch["orig_gap"][id(other)] if exact else None
            for e in other["elements"]:
                arc = element_arc(e) if e[0] == "arc" else None
                if arc is None:
                    out.append(Ob(LineString([e[1], e[-1]]), d, ref))
                else:
                    out.append(Ob(arc_band(arc, ARC_MAX_ERROR), d, ref, arc))
        return out

    def _report(self, ch, e):
        x, y = e.where if e.where else ch["orig"][0]
        self.messages.append(f"Net '{ch['net'] or '<no net>'}' on {self.board.get_layer_name(ch['layer'])} "
                             f"near ({x / 1e6:.3f}, {y / 1e6:.3f}) mm: left as is, {e}")

    @staticmethod
    def _rounded(elements, start, end) -> list[tuple]:
        """Elements on whole nm, neighbouring pieces sharing end points exactly. Raises
        RubberBandError if there would be a sliver."""
        r = lambda p: (int(round(p[0])), int(round(p[1])))
        out, cur = [], start
        for k, e in enumerate(elements):
            stop = end if k == len(elements) - 1 else r(e[-1])
            if stop == cur:
                continue
            mid = r(e[2]) if e[0] == "arc" else None
            if mid is not None and abs((stop[0] - cur[0]) * (mid[1] - cur[1]) - (stop[1] - cur[1]) * (mid[0] - cur[0])) \
                    < 1.5 * math.dist(cur, stop):
                mid = None      # mid < 1.5 nm off the chord: rounding would bend it any which way
            out.append(("arc", cur, mid, stop) if mid else ("line", cur, stop))
            cur = stop
        # A line of a few nm (an arc ending a hair before the run's end, which sits exactly
        # at the clearance) is folded into the piece before it.
        for k in range(len(out) - 1, 0, -1):
            if out[k][0] == "line" and math.dist(out[k][1], out[k][2]) < 5:
                out[k - 1] = (*out[k - 1][:-1], out[k][2])
                del out[k]
                if k < len(out):
                    out[k] = (out[k][0], out[k - 1][-1], *out[k][2:])
        if len(out) > 1 and out[0][0] == "line" and math.dist(out[0][1], out[0][2]) < 5:
            del out[0]
            out[0] = (out[0][0], start, *out[0][2:])
        for e in out:
            if len(out) > 1 and e[0] == "line" and math.dist(e[1], e[2]) < MIN_SEGMENT:
                raise RubberBandError(f"it would need a {math.dist(e[1], e[2]) / 1000:.1f} um straight piece", e[1])
        return out

    def run(self):
        selected = [i for i in self.board.get_selection(
            [KiCadObjectType.KOT_PCB_TRACE, KiCadObjectType.KOT_PCB_ARC])
            if isinstance(i, (Track, ArcTrack))]
        if not selected:
            self.messages.append("Select one or more track segments first.")
            return

        progress = getattr(self, "progress", None) or _NoProgress()
        progress.show("Reading the board…")
        self.load()
        progress.show("Preparing runs…")
        moving = {t.id.value for t in selected}
        chains = []
        for layer, net, width, items, pts in self.chains(selected):
            region = LineString(pts).convex_hull.buffer(2_000_000)
            raw = self.obstacles_for_layer(layer, net, region, moving)
            orig_line = LineString(pts)
            static = [Ob(g, d, orig_line.distance(g), o.circle)
                      for o, (g, d) in zip(raw, self.required(raw, net, width / 2))]
            elements = _item_elements(items, pts[0])
            chains.append(dict(layer=layer, net=net, width=width, items=items, orig=list(pts),
                               cur=list(pts), orig_elements=elements, elements=elements, new=False,
                               length=path_length(sample_elements(elements, 10.0)),
                               static=static, region=region, error=None, neighbours=[], orig_gap={}))

        # neighbours = other-net runs on the same layer that can affect this one
        if chains:
            tree = STRtree([ch["region"] for ch in chains])
            for i, ch in enumerate(chains):
                for j in tree.query(ch["region"], predicate="intersects"):
                    other = chains[int(j)]
                    if other is not ch and other["layer"] == ch["layer"] and other["net"] != ch["net"]:
                        ch["neighbours"].append(other)
                        ch["orig_gap"][id(other)] = LineString(ch["orig"]).distance(LineString(other["orig"]))

        # Each run is an obstacle for its neighbours: when one gets shorter, the runs next to
        # it are redone, until nothing changes. A change has to gain over 1 um, so this ends;
        # MAX_REDOS is only a safety net.
        # shortest first: an inner run of a bus then settles before the ones round it
        queue = deque(sorted((ch for ch in chains if len(ch["cur"]) >= 3), key=lambda ch: path_length(ch["orig"])))
        queued = {id(ch) for ch in queue}
        first, done = len(queue), 0
        was = sum(ch["length"] for ch in chains)
        now = was

        def lengths():
            return f"Total length {now / 1e6:.3f} mm (was {was / 1e6:.3f} mm, " \
                   f"{(now - was) / was * 100 if was else 0.0:+.1f} %)"
        while queue and done < MAX_REDOS * max(1, first):
            if progress.cancelled:
                self.messages.append("Cancelled; nothing was changed.")
                return
            ch = queue.popleft()
            queued.discard(id(ch))
            done += 1
            if done <= first:
                progress.show(f"Tightening run {done} of {first}…", done / first, lengths())
            else:
                progress.show(f"Runs next to ones that moved: {len(queue) + 1} to redo…",
                              done / (done + len(queue)), lengths())
            obstacles = ch["static"] + self._dynamic(ch)
            try:
                new_pts = pull_taut([(float(x), float(y)) for x, y in ch["cur"]], obstacles)
                ipts = [(int(round(x)), int(round(y))) for x, y in new_pts]
                ipts[0], ipts[-1] = ch["orig"][0], ch["orig"][-1]
                ipts = [p for i, p in enumerate(ipts) if i == 0 or p != ipts[i - 1]]
                # Go on even when the taut path is barely shorter (a 179 degree notch, a
                # curve made of segments): the exact path is built from it.
                if path_length(ipts) > path_length(ch["cur"]) + 2:     # longer, beyond rounding
                    continue
                if USE_ARCS:
                    exact = exact_path([(float(x), float(y)) for x, y in ipts],
                                       ch["static"] + self._dynamic(ch, exact=True))
                else:
                    exact = [("line", a, b) for a, b in zip(ipts, ipts[1:])]
                elements = self._rounded(exact, ch["orig"][0], ch["orig"][-1])
            except RubberBandError as e:
                # A redo that fails keeps the last result: neighbours that moved since were
                # fitted against it. Only a run that never worked is left as it was.
                if not ch["new"]:
                    ch["error"] = e
                continue
            # Accept it now, or not at all: neighbours are fitted against what this run is,
            # so that has to be what gets written.
            old = ch["elements"]
            length = path_length(sample_elements(elements, 10.0))
            exact_gain = ch["length"] - length
            more = len(elements) - len(old)
            fixed = min_margin(old, ch["static"]) < -VIOLATION_TOL <= min_margin(elements, ch["static"])
            if not (fixed or (more < 0 and exact_gain > -2) or (more <= 0 and exact_gain > 1000)
                    or exact_gain > BIG_GAIN):
                continue
            ch["cur"], ch["elements"], ch["new"], ch["error"] = ipts, elements, True, None
            now -= ch["length"] - length
            ch["length"] = length
            for nb in ch["neighbours"]:
                if id(nb) not in queued and len(nb["cur"]) >= 3:
                    queue.append(nb)
                    queued.add(id(nb))

        to_remove, to_create = [], []
        for ch in chains:
            if ch["error"] is not None:
                self._report(ch, ch["error"])
                continue
            if not ch["new"]:
                continue
            elements = ch["elements"]
            to_remove.extend(ch["items"])
            for el in elements:
                nt = ArcTrack() if el[0] == "arc" else Track()
                nt.start = Vector2.from_xy(*el[1])
                if el[0] == "arc":
                    nt.mid = Vector2.from_xy(*el[2])
                nt.end = Vector2.from_xy(*el[-1])
                nt.width = ch["width"]
                nt.layer = ch["layer"]
                nt.net = ch["items"][0].net
                to_create.append(nt)

        if not to_remove:
            if not self.messages:
                self.messages.append("Nothing to change: selection is already taut.")
            return

        # No begin_commit/push_commit: those requests carry no document, and KiCad 10 can
        # route them to the schematic editor's handler, which crashes KiCad. Without a
        # commit each request is its own undo step, so send exactly two: create the new
        # tracks first (a failure then leaves the old ones in place), then remove the old.
        if progress.cancelled:
            self.messages[:] = ["Cancelled; nothing was changed."]
            return
        progress.show("Writing tracks…", None, lengths())
        created = list(self.board.create_items(to_create))
        self.board.remove_items(to_remove)
        try:
            self.board.clear_selection()
            self.board.add_to_selection(created)
        except Exception:
            pass


class _NoProgress:
    cancelled = False

    def show(self, text, fraction=None, detail=None):
        pass

    def close(self):
        pass


def _wx_app():
    """The wx.App (wxPython ships with KiCad); None if wx is unavailable."""
    try:
        import wx
        return wx.GetApp() or wx.App(False)
    except Exception:
        return None


class _Progress(_NoProgress):
    """A small window with a progress bar and Cancel (wxPython; nothing if unavailable)."""

    def __init__(self):
        self.frame, self._last = None, 0.0
        try:
            import wx
            self.app = _wx_app()
            frame = wx.Frame(None, title="Rubber band",
                             style=wx.CAPTION | wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.STAY_ON_TOP)
            panel = wx.Panel(frame)
            box = wx.BoxSizer(wx.VERTICAL)
            self.label = wx.StaticText(panel, size=(360, -1), style=wx.ST_NO_AUTORESIZE)
            self.detail = wx.StaticText(panel, size=(360, -1), style=wx.ST_NO_AUTORESIZE)
            self.bar = wx.Gauge(panel, range=1000, size=(360, -1))
            cancel = wx.Button(panel, wx.ID_CANCEL, "Cancel")
            cancel.Bind(wx.EVT_BUTTON, self._cancel)
            frame.Bind(wx.EVT_CLOSE, self._cancel)
            box.Add(self.label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
            box.Add(self.detail, 0, wx.LEFT | wx.RIGHT | wx.TOP, 12)
            box.Add(self.bar, 0, wx.ALL, 12)
            box.Add(cancel, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)
            panel.SetSizer(box)
            box.Fit(frame)
            frame.Centre()
            frame.Show()
            self.frame = frame
        except Exception:
            self.frame = None

    def _cancel(self, _event=None):
        self.cancelled = True
        self.show("Cancelling…")

    def show(self, text, fraction=None, detail=None):
        """`detail`: a second line (the total length); None leaves it as it is."""
        if self.frame is None:
            return
        now = time.monotonic()
        if fraction is not None and 0 < fraction < 1 and now - self._last < 0.05:
            return
        self._last = now
        try:
            self.label.SetLabel(text)
            if detail is not None:
                self.detail.SetLabel(detail)
            if fraction is not None:
                self.bar.SetValue(int(round(fraction * 1000)))
            else:
                self.bar.Pulse()
            self.app.Yield(True)    # draw, and let Cancel be clicked
        except Exception:
            self.frame = None

    def close(self):
        if self.frame is not None:
            try:
                self.frame.Destroy()
                self.app.Yield(True)
            except Exception:
                pass
            self.frame = None


def _single_instance():
    """An exclusive lock held while this process runs (released when it exits, even on a
    crash), or None if another run has it: two runs at once would both write tracks."""
    f = open(os.path.join(tempfile.gettempdir(), "rubberband-tracks.lock"), "a+")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _notify(msg: str):
    print(msg, file=sys.stderr)
    try:
        import wx
        if _wx_app() is not None:
            wx.MessageBox(msg, "Rubber band", wx.OK | wx.ICON_INFORMATION)
    except Exception:
        pass


def main():
    lock = _single_instance()
    if lock is None:
        _notify("Rubber band is already running; wait for it to finish.")
        return
    progress = _Progress()
    try:
        rb = RubberBand()
        rb.progress = progress
        rb.run()
        progress.close()
        if rb.messages:
            _notify("\n".join(rb.messages))
    except Exception:
        progress.close()
        _notify("Rubber band failed:\n\n" + traceback.format_exc())
    finally:
        lock.close()


if __name__ == "__main__":
    main()
