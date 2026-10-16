#!/usr/bin/env python3
"""Build the Plugin and Content Manager zip: dist/rubberband-tracks-<version>.zip

Layout expected by KiCad's PCM ("Install from File"):
    metadata.json
    plugins/<plugin files>
    resources/icon.png
"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_FILES = [
    "plugin.json", "rubberband_action.py", "rubberband_core.py", "requirements.txt",
    "icon-light-24.png", "icon-light-48.png", "icon-dark-24.png", "icon-dark-48.png",
]


def main() -> int:
    meta = json.loads((ROOT / "pcm" / "metadata.json").read_text())
    plugin = json.loads((ROOT / "plugin" / "plugin.json").read_text())
    if meta["identifier"] != plugin["identifier"]:
        print("identifier mismatch between pcm/metadata.json and plugin/plugin.json", file=sys.stderr)
        return 1
    version = meta["versions"][0]["version"]

    out = ROOT / "dist" / f"rubberband-tracks-{version}.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "pcm" / "metadata.json", "metadata.json")
        z.write(ROOT / "pcm" / "icon.png", "resources/icon.png")
        for name in PLUGIN_FILES:
            z.write(ROOT / "plugin" / name, f"plugins/{name}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
