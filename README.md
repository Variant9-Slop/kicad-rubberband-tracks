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
- Output is exact: straight lines and arcs round vias, holes, pad corners (sharp or
  rounded), track ends and other arcs, at the required clearance. A run that can't be made
  clean is left as it was and listed in the popup with its location.
- Tracks that miss the clearance by a hair (up to 2 µm, as earlier versions of this plugin
  could leave them) are pushed out to the rule when you run it on them.

Settings are at the top of `plugin/rubberband_action.py`.

## Development

```
pip install -r requirements-dev.txt
pytest
```

## License

MIT
