# Rubber Band Tracks

KiCad plugin that pulls selected track segments taut: each selected run becomes the
shortest path that passes every obstacle on the same side as before.

Requires KiCad 9 or 10 (IPC API).

## Install

1. Preferences → Plugins → enable **KiCad API server**.
2. Build the package: `python scripts/build_pcm.py` → `dist/rubberband-tracks-<version>.zip`.
3. Plugin and Content Manager → **Install from File** → pick the zip.
4. Restart KiCad. The toolbar button appears once KiCad has set up the plugin's Python
   environment (shapely, kicad-python).

## Use

Select track segments (one net or many) and click the button. Undo takes two steps.
Run DRC and refill zones afterwards.

- Run ends stay fixed at pads, vias, branches, width changes and unselected tracks.
- Avoids other nets' tracks, pads and vias, drill holes, Edge.Cuts (including footprint
  outlines) and track keepouts. Copper zones are ignored; they refill.
- Clearances come from net classes and from the board's minimum, copper-to-hole and
  copper-to-edge rules in the `.kicad_pro` file. Custom DRC rules are not read.
- Never gets closer to anything than the rules require, or than the original track already was.
- A run that can't get shorter is still replaced if it comes out in fewer pieces: a
  straight line with a slight notch, or a curve drawn as many short segments.
- Bends around vias, holes, pad corners (sharp or rounded) and track ends become arcs at
  the required clearance plus 1 µm, joined by straight tangents. Where arcs would not fit,
  that run keeps straight segments (`USE_ARCS = False` turns arcs off).

Settings are at the top of `plugin/rubberband_action.py`.

## Development

```
pip install -r requirements-dev.txt
pytest
```

## License

MIT
