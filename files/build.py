"""Builds a Colab notebook for any video plan.

Usage:  python3 build.py <plan>.json <out>.ipynb

The engine lives in real Python files (engine/cells/*, engine/lib/*), not in strings:
  engine/cells/shared/   cells every notebook uses (log, setup, video, editing, music, render, QC ...)
  engine/cells/narrator/ cells for narrator-only plans (no "characters" in the plan)
  engine/cells/cast/     cells for plans with characters (voices per character, tones, continuity checks)
  engine/lib/            plain Python modules (validation, drawn shots, music library). They are copied into
                         the notebook's "lib" cell and imported, and they are unit-tested locally (tests/).
A cell file may contain placeholders: __PLAN_JSON__, __PROJECT__, __VOICE__, __TTS_LIB__, __LIB_FILES__.
Nothing about a particular video is written in the engine: everything per video is in the plan.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS_DIR, LIB_DIR = HERE / "engine" / "cells", HERE / "engine" / "lib"

ORDER = ["settings", "log", "install", "setup", "lib", "plan", "helpers", "prefetch", "narration", "timeline",
         "diagrams", "keyframes", "editlib", "logic", "bench", "clips", "music", "sfx", "render", "qc", "metadata", "preview"]
VARIANT = {"settings", "plan", "narration", "timeline", "metadata"}      # differ between narrator and cast plans


TOPIC_ORDER = ORDER[:ORDER.index("lib") + 1] + ["topic"] + ORDER[ORDER.index("lib") + 1:]


def profile(plan):
    return "cast" if plan.get("characters") else "narrator"


def build_topic():
    """The topic notebook: type a topic; research, script, images and plan are made in the notebook (topic cell)
    by local models, then the same tested cast-notebook cells render it."""
    reps = {"__PLAN_JSON__": "null", "__TTS_LIB__": tts_lib(), "__LIB_FILES__": json.dumps(lib_files(), indent=0)}
    head = (CELLS_DIR / "topic" / "settings_head.py").read_text().rstrip("\n")
    md = """# 🎓 Topic → educational video (free, local models)

**Type a topic in the first cell**, add your e-mail in `WIKI_CONTACT`, pick a **T4 GPU** (Runtime → Change runtime type), then **Runtime → Run all**.

What happens: Wikipedia facts (with sources) → a local open model (Qwen3, Apache-2.0) writes a beginner script that may only use those facts → every sentence is fact-checked (numbers, names, entailment) → licensed real images from Wikimedia Commons (public domain / CC0 / CC BY, credited) or drawn cards → Kokoro voice with a speech check → music, subtitles → `final.mp4` + `qc_report.json`. Nothing is paid; nothing unverified is approved.

Progress is saved after every stage (`stages/` on Drive): if Colab disconnects, **Run all** again and it carries on. Upload as **Private**; answer **Yes** to *Altered or synthetic content*."""
    cells = [("markdown", md, None)]
    for tag in TOPIC_ORDER:
        if tag == "topic":
            src = (CELLS_DIR / "topic" / "topic.py").read_text().rstrip("\n")
        else:
            src = cell_source(tag, "cast")
        if tag == "settings":
            src = head + "\n\n" + src.replace('"__PROJECT__"', "TOPIC_PROJECT").replace('"__VOICE__"', "NARRATOR_VOICE")
            src = src.replace('STORY_MODE  = "__STORY_MODE__"', "STORY_MODE  = STORY_MODE                  ")
            src = src.replace("LICENSE_STRICT = __LICENSE_STRICT__", "LICENSE_STRICT = LICENSE_STRICT")
            src = src.replace("__STRICT_OPEN_SOURCE__", "False")
        for k, v in reps.items():
            src = src.replace(k, v)
        cells.append(("code", src, tag))
    return cells


def cell_source(tag, prof):
    folder = prof if tag in VARIANT else "shared"
    return (CELLS_DIR / folder / f"{tag}.py").read_text().rstrip("\n")


def tts_lib():
    """The voice-model functions of the narrator narration cell (everything before its per-sentence loop)."""
    src = cell_source("narration", "narrator")
    marker = "FLAGGED = []"
    assert src.count(marker) == 1, "narrator narration cell changed: update tts_lib()"
    return src.split(marker)[0].split("\n", 1)[1]


STUDIO_ONLY = ("studio.py", "studio_run.py")     # one-time Character Studio code: not needed to render videos


def lib_files(studio=False):
    """engine/lib/*.py as a {file name: source} dict, embedded in the notebook's lib cell (video notebooks leave out
    the Studio-only modules; the Character Studio notebook gets everything)."""
    if not LIB_DIR.exists():
        return {}
    return {p.name: p.read_text() for p in sorted(LIB_DIR.glob("*.py")) if studio or p.name not in STUDIO_ONLY}


def header(plan, prof):
    nb = plan.get("notebook", {})
    project = plan.get("project") or plan.get("project_id") or "my_video_v1"
    cast = ", ".join(f"**{c['name']}** ({c['voice']})" for c in plan.get("characters", {}).values())
    kinds = sorted({s.get("recipe", "video") for s in plan["shots"]})
    return f'''# {nb.get("emoji", "🎬")} {nb.get("name", "AI Video Maker")}: "{plan["title"]}" ({nb.get("length", "about 1–2 min")})

**What this notebook does:** plan → checks (logic, facts, continuity) → voices with a speech check → AI video and drawn scenes ({", ".join(kinds)}) → licensed music and sound effects → subtitles → `final.mp4` on your Google Drive.
''' + (f"\n**Cast:** {cast}.\n" if cast else "") + f'''
**Which GPU?** Free **T4** works (turbo AI video, ~3–5 min per AI shot; drawn scenes need no GPU). **L4 / A100**: better, more consistent characters.

**How to run:** pick the GPU (**Runtime → Change runtime type**), then **Runtime → Run all**, and allow Google Drive access. Progress is saved after every step: if Colab disconnects, **Run all** again and it carries on.

**Output:** `MyDrive/ai-video/{project}/render/` → `final.mp4`, `subtitles.srt`, `script.txt`, `storyboard.md`, `validation_report.json`, `metadata.json`, `qc_report.json`. Run log (send it to Claude if anything fails): `.../logs/`.

**Rules:** stock voices only, never clone a real person. Upload as **Private** and answer **Yes** to *"Altered or synthetic content"*.'''


def build(plan):
    prof = profile(plan)
    reps = {"__PLAN_JSON__": json.dumps(plan, indent=1, ensure_ascii=False),
            "__PROJECT__": plan.get("project") or plan.get("project_id") or "my_video_v1",
            "__VOICE__": plan.get("characters", {}).get("narrator", {}).get("voice", "am_michael"),
            "__TTS_LIB__": tts_lib(),
            "__LIB_FILES__": json.dumps(lib_files(), indent=0),
            "__STORY_MODE__": plan.get("story_mode", "narrator"),
            "__LICENSE_STRICT__": str(bool(plan.get("license_strict", False))),   # story plans: strict by default
            "__STRICT_OPEN_SOURCE__": str(bool(plan.get("strict_open_source", False))),   # hosted Gemini opt-in only
            "__STORYBOOK_PACK__": json.dumps(storybook_pack_ref(plan))}
    cells = [("markdown", header(plan, prof), None)]
    for tag in ORDER:
        if tag == "lib" and not lib_files():
            continue
        src = cell_source(tag, prof)
        for k, v in reps.items():
            if k in src:
                src = src.replace(k, v)
        cells.append(("code", src, tag))
    return cells


def storybook_pack_ref(plan):
    """{"name", "sha256"} of the optional asset pack for this plan ({} for other plans). The pack itself is a
    separate file (write_storybook_pack): embedding 4–5 MB of image data in a notebook cell froze Colab's editor."""
    data = storybook_pack(plan)
    if not data:
        return {}
    import hashlib
    return {"name": f"storybook_assets_{hashlib.sha256(data).hexdigest()[:12]}.tar.gz",
            "sha256": hashlib.sha256(data).hexdigest()}


def write_storybook_pack(plan, out_dir):
    ref = storybook_pack_ref(plan)
    if ref:
        (Path(out_dir) / ref["name"]).write_bytes(storybook_pack(plan))
    return ref


def storybook_pack(plan):
    """tar.gz bytes of exactly the asset versions a storybook plan pins (colab/assets); b"" for other plans.
    Deterministic (sorted files, fixed timestamps), so the same assets always give the same sha256."""
    if not plan.get("visual_style_version"):
        return b""
    import io, tarfile, gzip
    root = HERE / "assets"
    dirs = [Path("characters") / c / a["version"] for c, a in plan.get("character_assets", {}).items()] + \
           [Path("environments") / e / v for e, v in plan.get("environments", {}).items()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for d in dirs:
            for f in sorted((root / d).rglob("*")):
                ti = tf.gettarinfo(str(f), arcname=str(d / f.relative_to(root / d)))
                ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
                if f.is_file():
                    with open(f, "rb") as fh:
                        tf.addfile(ti, fh)
                else:
                    tf.addfile(ti)
    return gzip.compress(buf.getvalue(), mtime=0)


def write_notebook(cells, path):
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for kind, src, tag in cells:
        lines = src.split("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
        if kind == "code":
            cell.update(execution_count=None, outputs=[], metadata={"tags": [tag]})
        nb["cells"].append(cell)
    json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
    print("wrote", path, len(nb["cells"]), "cells")


if __name__ == "__main__":
    if sys.argv[1] == "--topic":                    # python3 build.py --topic colab_topic_to_video.ipynb
        write_notebook(build_topic(), sys.argv[2])
    else:
        plan = json.load(open(sys.argv[1]))
        write_notebook(build(plan), sys.argv[2])
        ref = write_storybook_pack(plan, Path(sys.argv[2]).resolve().parent)
        if ref:
            print(f"optional asset pack {ref['name']}: upload it once to MyDrive/ai-video/_assets/ to skip drawing "
                  "the provisional characters in Colab (~6 min)")
