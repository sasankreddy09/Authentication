"""Reference and approval contracts for opt-in cinematic characters; no image model imports."""
import copy
import hashlib
import json
import shutil
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_review(path, files, kind, reviewer, approved=False):
    """Call only after the named person reviews actual files, never from an automatic score."""
    if approved is not True or not reviewer.strip() or not files:
        raise ValueError("Explicit human approval, reviewer and reviewed files are required")
    files = [Path(f) for f in files]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError("Review inputs are missing")
    import datetime
    data = {"kind": kind, "reviewer": reviewer, "approved": True,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "files": {str(p.resolve()): sha(p) for p in files}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return data


def require_review(path, files, kind):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Human {kind} review is pending: {path}")
    data = json.loads(path.read_text())
    if data.get("approved") is not True or data.get("kind") != kind or not data.get("reviewer"):
        raise ValueError(f"Invalid {kind} approval")
    required = {str(Path(p).resolve()) for p in files}
    if not required.issubset(data.get("files", {})):
        raise ValueError("Approval does not cover the current master")
    for name, expected in data["files"].items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError(f"Reviewed file changed or disappeared: {name}")
    return data


def preflight(references, output):
    """CPU-safe reference validation; report actual hardware without downloading a model."""
    from PIL import Image
    refs = {}
    for cid in ("arjun", "meera"):
        p = Path(references[cid])
        if not p.is_file():
            raise FileNotFoundError(f"Upload the supplied {cid} reference to {p}")
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            refs[cid] = {"file": str(p), "sha256": sha(p), "size": list(im.size)}
    import platform
    report = {"python": platform.python_version(), "references": refs, "gpu": None,
              "disk_free_gb": round(shutil.disk_usage(Path(output).parent).free / 2**30, 2)}
    try:
        import torch
        report.update(torch=torch.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report.update(gpu=p.name, vram_gb=p.total_memory / 2**30,
                          capability=[p.major, p.minor], native_bf16=p.major >= 8)
    except ImportError:
        report["torch"] = "not installed"
    try:
        import psutil
        report["ram_available_gb"] = psutil.virtual_memory().available / 2**30
    except ImportError:
        report["ram_available_gb"] = None
    Path(output).write_text(json.dumps(report, indent=2))
    return report


def validate_plan(plan, root):
    """New cinematic plans require approved, hash-pinned atlases. Legacy plans are unaffected."""
    if not plan.get("cinematic_characters"):
        return
    import storybook as SB
    if plan.get("visual_style_version") != "storybook_v1":
        raise ValueError("Cinematic characters require the Storybook renderer")
    pins = plan.get("character_assets", {})
    if not {"arjun", "meera"}.issubset(pins):
        raise ValueError("Both cinematic characters must be pinned")
    for cid, pin in pins.items():
        if pin.get("status") != "approved" or not pin.get("manifest_sha256") or not pin.get("content_sha256"):
            raise ValueError(f"{cid}: approved manifest and content pins required")
        ch = SB.load(cid, pin["version"], root, pin["manifest_sha256"], pin["content_sha256"])
        if ch.man.get("status") != "approved" or ch.man.get("asset_type") != "atlas_v1":
            raise ValueError(f"{cid}: approved cinematic atlas required; procedural art is not accepted")
        if not ch.man.get("provenance", {}).get("approved_by"):
            raise ValueError(f"{cid}: human artwork approval missing")


def bind_plan(plan, root, versions):
    """Create a new plan, preserving every story/audio field and the original plan."""
    import storybook as SB
    out = copy.deepcopy(plan)
    out["visual_style_version"] = "storybook_v1"
    out["cinematic_characters"] = True
    out["character_assets"] = {}
    for cid, version in versions.items():
        ch = SB.load(cid, version, root)
        out["character_assets"][cid] = {"version": version, "status": ch.man["status"],
            "art_version": ch.man["art_version"], "manifest_sha256": SB.manifest_sha(ch.dir / "manifest.json"),
            "content_sha256": SB.content_fingerprint(ch.man)}
    validate_plan(out, root)
    return out


RHUBARB = {"A": "closed", "B": "teeth", "C": "small", "D": "wide", "E": "round",
           "F": "round", "G": "teeth", "H": "small", "X": "rest"}


def rhubarb_visemes(cues, fps, frames):
    """Optional Rhubarb JSON to existing mouth IDs; does not replace audio-derived visemes."""
    import math
    if fps <= 0 or frames < 0:
        raise ValueError("Invalid timing")
    out = ["rest"] * frames
    last = 0.0
    for cue in cues:
        start, end = float(cue["start"]), float(cue["end"])
        if not math.isfinite(start + end) or start < last or end < start or cue["value"] not in RHUBARB:
            raise ValueError("Invalid or overlapping mouth cue")
        for i in range(max(0, math.ceil(start * fps)), min(frames, math.ceil(end * fps))):
            out[i] = RHUBARB[cue["value"]]
        last = end
    return out
