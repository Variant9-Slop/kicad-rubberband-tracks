"""Run the plugin on a real board offline and check the result with KiCad's DRC.

KiCad's pcbnew module loads the board, a stand-in for kipy's Board feeds it to the plugin,
and the result is written to a NEW file (with the project file copied next to it) that
kicad-cli checks, zones refilled. The input board is only read.

usage: python scripts/check_board.py <in.kicad_pcb> <out.kicad_pcb> [net names...]
       (all tracks are selected unless nets are given; run again on <out> for a 2nd pass)
Needs KiCad's pcbnew Python module (PCBNEW_PATH, default /usr/lib64/python3.14/site-packages)
and kicad-cli, plus the test requirements.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugin"))
sys.path.append(os.environ.get("PCBNEW_PATH", "/usr/lib64/python3.14/site-packages"))

import pcbnew  # noqa: E402
from kipy.board_types import (  # noqa: E402
    ArcTrack, BoardArc, BoardCircle, BoardLayer, BoardPolygon, BoardRectangle, BoardSegment,
    FootprintInstance, Net, Pad, Track, Via, Zone,
)
from kipy.geometry import PolygonWithHoles, PolyLine, PolyLineNode, Vector2  # noqa: E402
from kipy.proto.board import board_types_pb2 as bt  # noqa: E402
from kipy.project_types import NetClass  # noqa: E402

import rubberband_action  # noqa: E402


def kl(board, layer_id):
    """pcbnew layer id -> kipy BoardLayer."""
    return getattr(BoardLayer, "BL_" + board.GetLayerName(layer_id).replace(".", "_"))


def v2(p):
    return Vector2.from_xy(int(p.x), int(p.y))


def mknet(name):
    n = Net()
    n.name = name
    return n


def poly_to_proto(sps, idx=0):
    out = PolygonWithHoles()
    ol = PolyLine()
    o = sps.Outline(idx)
    for i in range(o.PointCount()):
        p = o.CPoint(i)
        ol.append(PolyLineNode.from_xy(int(p.x), int(p.y)))
    out.outline = ol
    return out


class RealBoard:
    def __init__(self, path):
        self.path = Path(path)
        self.b = pcbnew.LoadBoard(str(path))
        b = self.b
        self.created, self.removed = [], []
        self.tracks, self.vias, self.pads = [], [], []
        self.pcb_of = {}
        for t in b.GetTracks():
            c = t.GetClass()
            if c == "PCB_VIA":
                v = Via()
                v.proto.id.value = t.m_Uuid.AsString()
                v.position = v2(t.GetPosition())
                v.net = mknet(t.GetNetname())
                v.diameter = t.GetWidth(pcbnew.F_Cu)
                v.drill_diameter = t.GetDrillValue()
                v._layers = {kl(b, l) for l in range(pcbnew.PCB_LAYER_ID_COUNT)
                             if b.IsLayerEnabled(l) and pcbnew.IsCopperLayer(l) and t.FlashLayer(l)}
                self.vias.append(v)
            else:
                if c == "PCB_ARC":
                    k = ArcTrack()
                    k.mid = v2(t.GetMid())
                else:
                    k = Track()
                k.proto.id.value = t.m_Uuid.AsString()
                k.start, k.end = v2(t.GetStart()), v2(t.GetEnd())
                k.width = t.GetWidth()
                k.layer = kl(b, t.GetLayer())
                k.net = mknet(t.GetNetname())
                self.tracks.append(k)
            self.pcb_of[t.m_Uuid.AsString()] = t
        self.footprints = []
        for f in b.GetFootprints():
            fi = FootprintInstance()
            items = []
            for g in f.GraphicalItems():
                if g.GetClass() == "PCB_SHAPE" and g.GetLayer() == pcbnew.Edge_Cuts:
                    s = self.shape(g)
                    if s is not None:
                        items.append(s)
            fi.definition.items = items
            self.footprints.append(fi)
            for p in f.Pads():
                k = Pad()
                k.proto.id.value = p.m_Uuid.AsString()
                k.position = v2(p.GetPosition())
                k.net = mknet(p.GetNetname())
                d = p.GetDrillSize()
                k.proto.pad_stack.drill.diameter.x_nm, k.proto.pad_stack.drill.diameter.y_nm = d.x, d.y
                k.proto.pad_stack.angle.value_degrees = p.GetOrientationDegrees()
                k._polys, k._layers = {}, set()
                for l in range(pcbnew.PCB_LAYER_ID_COUNT):
                    if b.IsLayerEnabled(l) and pcbnew.IsCopperLayer(l) and p.FlashLayer(l):
                        sps = pcbnew.SHAPE_POLY_SET()
                        p.TransformShapeToPolygon(sps, l, 0, b.GetDesignSettings().m_MaxError, pcbnew.ERROR_INSIDE)
                        if sps.OutlineCount():
                            k._polys[kl(b, l)] = poly_to_proto(sps)
                            k._layers.add(kl(b, l))
                cl = k.proto.pad_stack.copper_layers[0]
                shape = {pcbnew.PAD_SHAPE_CIRCLE: bt.PSS_CIRCLE, pcbnew.PAD_SHAPE_RECTANGLE: bt.PSS_RECTANGLE,
                         pcbnew.PAD_SHAPE_OVAL: bt.PSS_OVAL, pcbnew.PAD_SHAPE_ROUNDRECT: bt.PSS_ROUNDRECT}
                cl.shape = shape.get(p.GetShape(pcbnew.F_Cu), bt.PSS_CUSTOM)
                sz, off = p.GetSize(pcbnew.F_Cu), p.GetOffset(pcbnew.F_Cu)
                cl.size.x_nm, cl.size.y_nm = sz.x, sz.y
                cl.offset.x_nm, cl.offset.y_nm = off.x, off.y
                cl.corner_rounding_ratio = p.GetRoundRectRadiusRatio(pcbnew.F_Cu)
                self.pads.append(k)
        self.shapes = []
        for d in b.GetDrawings():
            if d.GetClass() == "PCB_SHAPE":
                s = self.shape(d)
                if s is not None:
                    self.shapes.append(s)
        self.zones = []
        for z in b.Zones():
            if z.GetIsRuleArea() and z.GetDoNotAllowTracks():
                k = Zone()
                k.proto.type = bt.ZT_RULE_AREA
                k.proto.rule_area_settings.keepout_tracks = True
                for l in range(pcbnew.PCB_LAYER_ID_COUNT):
                    if z.IsOnLayer(l) and pcbnew.IsCopperLayer(l) and b.IsLayerEnabled(l):
                        k.proto.layers.append(kl(b, l))
                k.proto.outline.polygons.append(poly_to_proto(z.Outline()).proto)
                self.zones.append(k)
        self.selection = []

    def shape(self, g):
        st = g.GetShape()
        lay = kl(self.b, g.GetLayer())
        if st == pcbnew.SHAPE_T_SEGMENT:
            s = BoardSegment()
            s.start, s.end = v2(g.GetStart()), v2(g.GetEnd())
        elif st == pcbnew.SHAPE_T_ARC:
            s = BoardArc()
            s.start, s.mid, s.end = v2(g.GetStart()), v2(g.GetArcMid()), v2(g.GetEnd())
        elif st == pcbnew.SHAPE_T_CIRCLE:
            s = BoardCircle()
            s.center, s.radius_point = v2(g.GetCenter()), v2(g.GetEnd())
        elif st == pcbnew.SHAPE_T_POLY:
            s = BoardPolygon()
            sps = g.GetPolyShape()
            for i in range(sps.OutlineCount()):
                s.proto.shape.polygon.polygons.append(poly_to_proto(sps, i).proto)
        elif st == pcbnew.SHAPE_T_RECTANGLE:
            s = BoardRectangle()
            s.top_left, s.bottom_right = v2(g.GetStart()), v2(g.GetEnd())
        else:
            print("skipping shape type", st)
            return None
        s.layer = lay
        return s

    # --- kipy Board API used by the plugin
    def get_selection(self, types=None): return self.selection
    def get_tracks(self): return self.tracks
    def get_vias(self): return self.vias
    def get_pads(self): return self.pads
    def get_zones(self): return self.zones
    def get_shapes(self): return self.shapes
    def get_footprints(self): return self.footprints
    def get_layer_name(self, layer): return BoardLayer.Name(layer)

    def get_project(self):
        return types.SimpleNamespace(path=str(self.path.parent), name=self.path.stem)

    def get_nets(self):
        return [mknet(n) for n in sorted({i.net.name for i in self.tracks + self.vias + self.pads})]

    def get_netclass_for_nets(self, nets):
        out = {}
        for n in nets:
            c = self.b.GetDesignSettings().m_NetSettings.GetEffectiveNetClass(n.name)
            nc = NetClass()
            nc.proto.name = c.GetName()
            nc.clearance = c.GetClearance()
            out[n.name] = nc
        return out

    def check_padstack_presence_on_layers(self, items, layer):
        return {i: {layer: layer in i._layers} for i in items}

    def get_pad_shapes_as_polygons(self, pads, layer):
        if isinstance(pads, list):
            return [p._polys.get(layer) for p in pads]
        return pads._polys.get(layer)

    def create_items(self, items):
        items = list(items)
        self.created.extend(items)
        return items

    def remove_items(self, items):
        self.removed.extend(items)

    def clear_selection(self): pass
    def add_to_selection(self, items): pass

    # --- write the result into the pcbnew board and save a copy
    def apply_and_save(self, out):
        b = self.b
        lid = {b.GetLayerName(l): l for l in range(pcbnew.PCB_LAYER_ID_COUNT)}
        for k in self.removed:
            b.Remove(self.pcb_of[k.id.value])
        for k in self.created:
            t = pcbnew.PCB_ARC(b) if isinstance(k, ArcTrack) else pcbnew.PCB_TRACK(b)
            t.SetStart(pcbnew.VECTOR2I(k.start.x, k.start.y))
            if isinstance(k, ArcTrack):
                t.SetMid(pcbnew.VECTOR2I(k.mid.x, k.mid.y))
            t.SetEnd(pcbnew.VECTOR2I(k.end.x, k.end.y))
            t.SetWidth(k.width)
            t.SetLayer(lid[BoardLayer.Name(k.layer)[3:].replace("_", ".")])
            t.SetNet(b.FindNet(k.net.name))
            b.Add(t)
        pcbnew.SaveBoard(str(out), b)
        shutil.copy(self.path.with_suffix(".kicad_pro"), Path(out).with_suffix(".kicad_pro"))


def drc(pcb):
    rep = Path(pcb).with_suffix(".drc.json")
    subprocess.run(["kicad-cli", "pcb", "drc", "--refill-zones", "--format", "json", "--severity-error", "-o", str(rep), str(pcb)],
                   capture_output=True)
    data = json.loads(rep.read_text())
    out = []
    for v in data.get("violations", []):
        if v["type"] in ("clearance", "hole_clearance", "copper_edge_clearance", "tracks_crossing", "shorting_items"):
            out.append((v["type"], v["description"], [i.get("description", "")[:70] for i in v.get("items", [])],
                        [(round(i["pos"]["x"], 4), round(i["pos"]["y"], 4)) for i in v.get("items", [])][:1]))
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.resolve() == src.resolve() or dst.with_suffix(".kicad_pro").resolve() == src.with_suffix(".kicad_pro").resolve():
        sys.exit("refusing to overwrite the input board or its project file")
    rb_board = RealBoard(src)
    sel = sys.argv[3:] or ["all"]
    rb_board.selection = [t for t in rb_board.tracks if sel == ["all"] or t.net.name in sel]
    rb = rubberband_action.RubberBand.__new__(rubberband_action.RubberBand)
    rb.board, rb.messages = rb_board, []
    t0 = time.time()
    rb.run()
    print(f"plugin: {time.time() - t0:.1f} s, created {len(rb_board.created)}, removed {len(rb_board.removed)}")
    for m in rb.messages:
        print("  MSG", m)
    rb_board.apply_and_save(dst)
    for v in drc(dst):
        print("  DRC", v)
