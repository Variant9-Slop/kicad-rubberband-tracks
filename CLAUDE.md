# CLAUDE.md

KiCad IPC-API action plugin ("Rubber Band Tracks"): replaces selected track runs with the
shortest path of the same topology (same side of every obstacle).

## Layout

- `plugin/` — exactly what KiCad loads. Flat on purpose: `plugin.json` entrypoint and icon
  paths are relative to this folder, and the entrypoint imports `rubberband_core` as a sibling.
  - `rubberband_core.py` — pure geometry, no KiCad imports. `pull_taut(path, obstacles)`
    finds the topology, `exact_path(taut, obstacles)` the exact lines and arcs.
  - `rubberband_action.py` — KiCad side: reads the board, builds runs ("chains") and
    obstacles, runs passes, writes tracks back. Tunables at the top.
- `pcm/` — `metadata.json` and icon for the Plugin and Content Manager package.
- `scripts/build_pcm.py` — builds `dist/rubberband-tracks-<version>.zip`.
- `tests/` — pytest; `mock_board.py` fakes kipy's `Board` so the plugin runs without KiCad.

## Commands

- Tests: `pytest` (needs `kicad-python`, `shapely`, `numpy`, `pytest`).
- Package: `python scripts/build_pcm.py`.
- Debug in KiCad: `KICAD_ENABLE_WXTRACE=1 WXTRACE=KICAD_API kicad`.

## Algorithm (core)

A shortest path that keeps clear of obstacles is straight lines plus arcs round obstacle
features (via/hole centre, pad or polygon corner, rounded-corner centre, track end, arc
centre) at the required distance. Polygons are only used to find *which* features the
path wraps; the output is computed exactly. Never fit or guess curves from points: that
approach (v0.6.0–0.6.3 `fit_arcs`) produced slivers and sub-µm violations with every fix.

Obstacles are `Ob(geom, d, ref=None, arc=None)`: `geom` must stay `d` from the path;
`ref` = the original track's distance (default: the current path's); `arc` = true circle
and extent when `geom` is a band standing in for an arc.

1. `pull_taut`: grow each obstacle by `min(d, ref - eps)` (polygon scaled up by
   `_BUF_SCALE`), free space = convex hull of the path (+1 mm) minus grown obstacles;
   constrained Delaunay triangulation (no Steiner points, so loops in the triangle graph
   only go around obstacles). Sleeve = triangles the original path crosses, in order,
   immediate back-and-forth cancelled (do not remove all repeats; that drops real detours
   around obstacles). Funnel through the sleeve → taut polyline.
2. `exact_path`: each polyline corner → the obstacle whose grown outline it lies on,
   inside the bend → that obstacle's feature (nearest skeleton vertex, or the arc's centre
   if beside the arc). Consecutive corners on one feature = one wrap, radius = feature
   radius + `keep` (+ `ROUNDING` spare, dropped if that doesn't fit). Exact tangents
   between wraps (`_tangent`; nearly touching circles switch where they touch or cross,
   never via a µm sliver). Check against the true shapes (arcs sampled to 0.1 nm,
   `CHECK_TOL` 5 nm): add the feature a piece cuts into, drop a wrap whose sweep went
   negative, repeat. Arcs under `ARC_MIN_SAGITTA` (3 nm) off their chord become a corner.
3. `keep` = `d` if the original was within `SNAP_TOL` (2 µm) of it, else the original's
   distance (never tighter than the original); capped by the fixed ends' distance. So
   re-running on tracks a hair too close (older versions) puts them right.

Plugin: pads are exact shapes from the pad stack (circle/oval/rect/roundrect: skeleton +
corner radius), used only if they match KiCad's polygon (Hausdorff ≤ 2·`PAD_POLY_ERROR`),
else that polygon. Arcs on the board are bands (`arc_band`: chords inside, tangents
outside) plus their true circle. Neighbouring selected runs are passed to each other as
their exact elements. A run is written if shorter, in fewer pieces, or fixing a
violation; if it can't be made clean (incl. a straight piece under `MIN_SEGMENT`) it is
left alone and reported. `USE_ARCS = False` writes the taut polyline instead (coarse:
can be ~0.07 % of the distance inside a capped obstacle).

Units are nanometres throughout (KiCad's internal unit).

## Constraints and gotchas

- Do not use `board.begin_commit()` / `push_commit()`: `BeginCommit` carries no document,
  and KiCad 10 can route it to the schematic editor's handler, which crashes KiCad
  (`API_HANDLER_EDITOR::checkForBusy` in `_eeschema.kiface`). Writes are two requests
  instead: create new tracks, then remove old ones (two undo steps).
- kipy's default request timeout is 2 s; the client is created with `API_TIMEOUT_MS`.
- Shapely buffers are polygons whose edges fall inside the true offset; `_BUF_SCALE`
  compensates (GEOS steps can be ~1.5× the nominal angle — measured, don't lower it).
- Board rules (hole/edge/min clearance) aren't exposed by the KiCad 10 API; they are read
  from `<project>.kicad_pro` → `board.design_settings.rules` (mm).
- Keep the author as "Variant" in `pcm/metadata.json` and `LICENSE`.

## Known issues

- "internal: result is longer than the original": one cause found and fixed (an endpoint
  lying exactly on a triangle edge, see `test_start_on_triangle_edge`). It was also seen on
  multi-net runs that may have another cause; `_nudge_out` was added for those, unverified.
- Custom DRC rules and copper zones are not considered.
- Custom pads (and trapezoid/chamfered) use KiCad's polygon: a path round their curved
  parts wraps each polygon corner, which can give straight pieces under `MIN_SEGMENT`
  (the run is then left alone and reported).
