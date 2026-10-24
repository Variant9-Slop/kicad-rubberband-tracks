# CLAUDE.md

KiCad IPC-API action plugin ("Rubber Band Tracks"): replaces selected track runs with the
shortest path of the same topology (same side of every obstacle).

## Layout

- `plugin/` — exactly what KiCad loads. Flat on purpose: `plugin.json` entrypoint and icon
  paths are relative to this folder, and the entrypoint imports `rubberband_core` as a sibling.
  - `rubberband_core.py` — pure geometry, no KiCad imports. `pull_taut(path, obstacles)`,
    `fit_arcs(path, obstacles, margin)`.
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

1. Each obstacle is `(geometry, required_centreline_distance[, reference_distance])`.
   It is grown by `min(required, reference - eps)`, reference defaulting to the path's
   current distance. So the result is never tighter than the rules nor than the original.
2. Free space = convex hull of the path (+1 mm) minus grown obstacles; constrained
   Delaunay triangulation (no Steiner points, so loops in the triangle graph only go
   around obstacles).
3. Sleeve = triangles the original path crosses, in order, with immediate back-and-forth
   cancelled (do not remove all repeats; that drops real detours around obstacles).
4. Funnel algorithm through the sleeve → shortest path. Final check against obstacles.
5. On write-out only (passes work on point lists), `fit_arcs`: each corner of the taut
   path is matched to the obstacle on the inside of its bend (preferring the previous
   corner's); consecutive corners around the same point, or around obstacle points that
   form a polygonised circle (≥4 points, ~exactly on it; later points may join within
   `ARC_JOIN_TOL`), become one arc at clearance + `ARC_MARGIN`, joined by exact tangents.
   A fitted circle that then breaks (no tangent, sweep mismatch, clearance) is split back
   into per-point arcs. The plugin tries margin, then 0, then arcs kept inside the
   segments (runs squeezed between selected neighbours). If none fits, or the result has a
   straight piece under `MIN_SEGMENT`, the run is left as it was and reported: never write
   the intermediate polyline (it is as fine as the polygons, i.e. slivers).
   Arcs under `ARC_MIN_SAGITTA` (3 nm) off their chord would be warped by rounding to nm;
   they become one corner where the tangents meet (bends of under ~0.5°).

Curves on the board (arc tracks, graphic arcs, circles) are obstacles as a thin band
between chords inside and tangents outside the true curve (`_arc_band`), so clearance to
the band is never more than to the curve, from either side. Their true circle and extent
go to `fit_arcs` as `circles`: corners around them use that centre and radius (never
re-fitted, never split), and the final check measures the true curve, so each ring of
tracks around the last doesn't creep outwards and hit the next one at minimum pitch.
Selected runs' own arcs are read as plain chords (`_track_points`).

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
- Where the original sits at exactly the required clearance, growth is capped at that
  distance and the polygon sides fall ~0.03 % inside it, so results can be ~0.15 µm under
  the rule (below KiCad's default 0.5 µm DRC epsilon).
