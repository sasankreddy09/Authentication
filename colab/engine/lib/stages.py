"""Stage checkpoints: a stage's result is saved with a hash of everything that shaped it, so a rerun (after a Colab
disconnect, or with unchanged settings) reuses it instead of doing the work again; a changed input re-runs it.

    sheet = stage(DIR, "research", {"topic": t, "v": 1}, lambda: do_research())
"""
from __future__ import annotations

import hashlib, json, time
from pathlib import Path


def key_of(inputs) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stage(folder, name, inputs, fn, log=print):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    f, k = folder / f"{name}.json", key_of(inputs)
    if f.exists():
        try:
            d = json.loads(f.read_text())
            if d.get("key") == k:
                log(f"stage {name}: reused (saved {d.get('saved')}, key {k})")
                return d["result"]
        except Exception:
            log(f"stage {name}: checkpoint unreadable, running again")
    t0 = time.time()
    result = fn()
    tmp = f.with_suffix(".tmp")                      # write-then-rename: an interrupted save never looks complete
    tmp.write_text(json.dumps({"key": k, "inputs": inputs, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "seconds": round(time.time() - t0, 1), "result": result}, indent=1, default=str))
    tmp.replace(f)
    log(f"stage {name}: done in {time.time() - t0:.0f}s (key {k})")
    return result
