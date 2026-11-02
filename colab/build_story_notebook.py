"""Kept so old commands still work. The engine now lives in engine/ and every notebook is built by build.py:

    python3 build.py <plan>.json <out>.ipynb

(narrator-only or with characters is decided by the plan itself: plans with "characters" get the cast cells).
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build

if __name__ == "__main__":
    build.write_notebook(build.build(json.load(open(sys.argv[1]))), sys.argv[2])
