# CLAUDE.md

KiCad IPC-API action plugin ("Rubber Band Tracks"): replaces selected track runs with the
shortest path of the same topology (same side of every obstacle).

## Layout

- `plugin/` — exactly what KiCad loads. Flat on purpose: `plugin.json` entrypoint and icon
  paths are relative to this folder, and the entrypoint imports `rubberband_core` as a sibling.
  - `rubberband_core.py` — pure geometry, no KiCad imports. `pull_taut(path, obstacles)`.
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

Units are nanometres throughout (KiCad's internal unit).

## Constraints and gotchas

- Plugin identifier must have ≥3 dot-separated parts (`local.hobby.rubberband-tracks`),
  or KiCad silently rejects it ("identifier ... does not meet requirements").
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

- "internal: result is longer than the original" still reported on some multi-net runs;
  not reproduced in tests yet. `_nudge_out` was added as a candidate fix, unverified.
- Custom DRC rules and copper zones are not considered.
