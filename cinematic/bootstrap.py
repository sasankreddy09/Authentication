"""Recover the existing notebook engine without changing unrelated source files."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "files/colab_three_d_friends_te_v3_storybook.ipynb"


def sources(studio=False):
    nb = json.loads(BASE.read_text())
    cells = ["".join(c.get("source", [])) for c in nb["cells"]]
    lib = next(s for s in cells if "# ==== 3b. Engine library" in s)
    match = re.search(r'json.loads\(r?"""(.*?)"""\)', lib, re.S)
    modules = json.loads(match.group(1))
    for p in (ROOT / "cinematic/lib").glob("*.py"):
        if studio or p.name not in ("studio.py", "studio_run.py"):
            modules[p.name] = p.read_text()
    return modules


def materialize(destination, studio=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources(studio).items():
        if Path(name).name != name:
            raise ValueError("Invalid embedded module path")
        (destination / name).write_text(source)
    return destination
