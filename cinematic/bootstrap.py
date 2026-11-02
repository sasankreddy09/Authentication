"""Load the canonical, editable Colab engine without importing model runtimes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sources(studio=False):
    return {p.name: p.read_text() for p in sorted((ROOT / "colab/engine/lib").glob("*.py"))
            if studio or p.name not in ("studio.py", "studio_run.py")}


def materialize(destination, studio=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in sources(studio).items():
        (destination / name).write_text(source)
    return destination
