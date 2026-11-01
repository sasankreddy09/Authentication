"""Bind approved cinematic characters into a NEW existing-production notebook."""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from bootstrap import materialize, sources


def build_video(template, plan, root, versions, output):
    import cinematic_workflow as CW
    bound = CW.bind_plan(plan, root, versions)
    nb = json.loads(Path(template).read_text())
    lib_count = plan_count = 0
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "# ==== 3b. Engine library" in src:
            # Replace only character modules; retain this video's unrelated engine version.
            match = re.search(r'json.loads\(r?"""(.*?)"""\)', src, re.S)
            modules = json.loads(match.group(1))
            for name in ("storybook.py", "storybook_scene.py", "cinematic_workflow.py"):
                modules[name] = sources()[name]
            src = src[:match.start(1)] + json.dumps(modules) + src[match.end(1):]
            lib_count += 1
        if "PLAN_EMBEDDED = json.loads" in src:
            match = re.search(r'PLAN_EMBEDDED = json.loads\(r?"""(.*?)"""\)', src, re.S)
            src = src[:match.start(1)] + json.dumps(bound, ensure_ascii=False) + src[match.end(1):]
            # A retained Drive plan must not silently reactivate provisional art.
            guard = '\nif PLAN.get("character_assets") != PLAN_EMBEDDED["character_assets"] or not PLAN.get("cinematic_characters"):\n    raise RuntimeError("Drive plan selects different artwork. Save the newly pinned plan to this project before running.")\nimport cinematic_workflow as CW\nos.environ.setdefault("STORYBOOK_ASSETS", str(Path(DRIVE_ROOT) / "_assets"))\nCW.validate_plan(PLAN, __import__("storybook").library_root())\n'
            anchor = 'STYLE, NEG ='
            if anchor not in src: raise ValueError("Unsupported plan cell")
            at = src.index(anchor)
            src = src[:at] + guard + src[at:]
            plan_count += 1
        cell["source"] = src.splitlines(True)
        if cell["cell_type"] == "code": cell.update(outputs=[], execution_count=None)
    if (lib_count, plan_count) != (1, 1):
        raise ValueError("Expected exactly one engine and one embedded production plan")
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    return bound


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True); ap.add_argument("--plan", required=True)
    ap.add_argument("--assets", required=True); ap.add_argument("--version", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0, str(materialize(tmp)))
        build_video(args.template, json.loads(Path(args.plan).read_text()), args.assets,
                    {c: args.version for c in ("arjun", "meera")}, args.output)
