"""Build colab/colab_character_studio.ipynb: the one-time Character Studio (Colab T4) for Arjun and Meera.

Usage:  python3 colab/build_studio.py colab/colab_character_studio.ipynb

Stages (each resumable; finished images are skipped on re-run): settings → setup → preflight → proof of concept →
APPROVAL GATE (you) → pose atlas → unload → cut-outs → atlas source (review) → import + previews → approval
(colab/approve_character.py) → optional environment plates. Normal video notebooks never load these models.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

MD = """# 🎨 Character Studio: cinematic Arjun and Meera (one-time, free Colab T4)

**Generate once. Approve once per version. Reuse in every video.**

1. **Proof of concept**: portrait, full-body master, one expression and one three-quarter view per character
   (FLUX.2 [klein] 4B, Apache-2.0). Look at the contact sheet next to your reference images.
2. **Your approval** (`POC_APPROVED` in the settings): face, hair, hoodie / butterflies, style, consistency.
   Nothing past this point runs until every item is `True`.
3. **Pose atlas**: 39 images per character, all made FROM the approved master (13 poses, 3D-glasses and
   popcorn variants, ¾ and back views, blink, mouth shapes, 11 expressions) → BiRefNet cut-outs (MIT) →
   an atlas version in `MyDrive/ai-video/_assets/` → previews → `approve_character.py`.

Why a pose atlas and not a cut-out rig: rotating cut-out limbs of this artwork breaks the shoulders and the shading
(see `reports/character_poc/approach_A_cutout_arm_test.png`); whole pose images keep it intact.

⚠️ Written and CPU-tested without a GPU. The first T4 run IS the benchmark (`studio/benchmark.json`); if a step
runs out of memory, set `SIZE = 640` and re-run that cell — finished images are never redone."""

CELLS = [
    ("settings", '''# ==== 1. Settings ====
STUDIO_CHARACTERS = ["arjun", "meera"]
BACKEND = "local"                # "local": this Colab GPU | "hosted": the official HF Space (small free quota, HF_TOKEN secret)
SIZE = 768                       # 768 x 1024 masters; 640 if memory runs out
STYLE_REFERENCES = {}            # art-direction images on Drive, e.g. {"arjun": ["/content/drive/MyDrive/refs/boy.png"]}
                                 # (guidance for rendering quality / lighting only — the characters stay original)
POC_APPROVED = {                 # YOUR approval after looking at studio/poc/contact_sheet.png (all must be True)
    "arjun_face": False, "arjun_hair": False, "arjun_golden_hoodie": False,
    "meera_face": False, "meera_butterfly_hairstyle": False, "shared_style": False, "visual_consistency": False}
POC_PICK = {"arjun": "full_body_master", "meera": "full_body_master"}   # which PoC image becomes the atlas master
NEW_VERSION = "v2"               # the atlas version folder (v1 = the code-drawn provisional characters)
MAKE_ENV_PLATES = False          # optional: generated backgrounds as environment v2 (same camera presets)
LICENSE_STRICT = True'''),
    ("setup", '''# ==== 2. Drive, packages, logging ====
import os, sys, json, time, shutil, logging, subprocess
from pathlib import Path
try:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT = "/content/drive/MyDrive/ai-video"
except Exception:
    DRIVE_ROOT = os.environ.get("TEST_DRIVE", "/tmp/ai-video")
ASSETS = Path(DRIVE_ROOT) / "_assets"; STUDIO = ASSETS / "studio"; STUDIO.mkdir(parents=True, exist_ok=True)
WORK = Path(os.environ.get("WORK_DIR", "/content/work")) / "studio"; WORK.mkdir(parents=True, exist_ok=True)
os.environ["STORYBOOK_ASSETS"] = str(ASSETS)
logging.basicConfig(filename=str(STUDIO / "studio.log"), level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
def log(m, *a):
    print(m); logging.info(m)
try:                                        # HF token (Colab Secrets) only for the hosted backend's quota; never printed
    from google.colab import userdata
    if userdata.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
pkgs = ["git+https://github.com/huggingface/diffusers", "bitsandbytes", "accelerate", "psutil"] if BACKEND == "local" else ["gradio_client", "psutil"]
r = subprocess.run([sys.executable, "-m", "pip", "-q", "install", "-U", *pkgs], capture_output=True, text=True)
log(f"pip {pkgs}: exit {r.returncode} {r.stderr[-500:]}")'''),
    ("lib", None),
    ("preflight", '''# ==== 3. Preflight: GPU, RAM, disk, dependencies, licences ====
import registry, studio, studio_run, storybook as SB, storybook_atlas as SA
BENCH = {"stage_seconds": {}}
REP = studio.gpu_report() if BACKEND == "local" else {"backend": "hosted"}
REP["disk_free_gb"] = round(shutil.disk_usage("/content" if os.path.exists("/content") else "/").free / 2 ** 30, 1)
log(f"preflight: {REP}")
for mid in (studio.FLUX_MODEL, studio.SEG_MODEL):
    ok, why = registry.model_ok(mid, strict=LICENSE_STRICT)
    log(("✅ " if ok else "❌ ") + why); assert ok, why
if BACKEND == "local":
    assert REP.get("cuda"), "no GPU: Runtime → Change runtime type → T4 GPU (or BACKEND = 'hosted')"
    assert REP["disk_free_gb"] > 25, "need ~25 GB free disk for the model download (transformer 7.8 GB + text encoder 8 GB + cache)"
    import diffusers; log(f"diffusers {diffusers.__version__}, Flux2KleinPipeline: {hasattr(diffusers, 'Flux2KleinPipeline')}")
BENCH["preflight"] = REP'''),
    ("poc", '''# ==== 4. Proof of concept (portrait, full-body master, expression, ¾ view) → contact sheet ====
t0 = time.time()
GEN = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log) if BACKEND == "local" else studio_run.HostedGen(log)
BENCH["model_load_s"] = getattr(GEN, "load_s", None)
POC = STUDIO / "poc"
try:
    st = studio_run.run_poc(GEN, POC, STUDIO_CHARACTERS, style_refs=STYLE_REFERENCES, log=log)
except studio_run.QuotaExhausted as e:
    st = {}; log(f"hosted quota exhausted: {e} — finished images kept; re-run this cell later")
from PIL import Image
items = [(f"{p.parent.name} {p.stem}", Image.open(p)) for p in sorted(POC.glob("*/*.png")) if ".partial" not in p.name]
for cid, refs in STYLE_REFERENCES.items():
    items += [(f"{cid} REFERENCE {Path(r).stem}", Image.open(r)) for r in refs]
studio.contact_sheet(items, POC / "contact_sheet.png", cols=4, cell=420)
BENCH["stage_seconds"]["poc"] = round(time.time() - t0, 1)
print(f"Look at {POC / 'contact_sheet.png'} (and the single images), then set POC_APPROVED in cell 1.")'''),
    ("gate", '''# ==== 5. APPROVAL GATE: nothing below runs until you approved the proof of concept ====
missing = [k for k, v in POC_APPROVED.items() if v is not True]
assert not missing, f"not approved yet: {missing} — look at the contact sheet and set POC_APPROVED in cell 1"
for cid in STUDIO_CHARACTERS:
    d = STUDIO / cid / "atlas"; d.mkdir(parents=True, exist_ok=True)
    src = POC / cid / f"{POC_PICK[cid]}.png"
    if not (d / "idle_front.png").exists():
        shutil.copy(src, d / "idle_front.png"); shutil.copy(src.with_suffix(".json"), d / "idle_front.json")
json.dump({"approved": POC_APPROVED, "pick": POC_PICK, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
          open(STUDIO / "poc_approval.json", "w"), indent=1)
log(f"proof of concept approved: {POC_APPROVED}")'''),
    ("atlas", '''# ==== 6. Pose atlas: 38 edits per character from the approved master (resumable) ====
t0 = time.time()
for cid in STUDIO_CHARACTERS:
    try:
        st = studio_run.run_atlas(GEN, STUDIO, cid, size=(SIZE, int(SIZE * 4 / 3)), log=log)
        log(f"{cid}: {st}")
    except studio_run.QuotaExhausted as e:
        log(f"quota exhausted: {e} — re-run this cell later"); break
BENCH["stage_seconds"]["atlas"] = round(time.time() - t0, 1)
if BACKEND == "local":
    GEN.unload(); log("image model released (one model family on the GPU at a time)")'''),
    ("cutout", '''# ==== 7. Cut-outs (BiRefNet, MIT) for every pose / view image ====
t0 = time.time()
for cid in STUDIO_CHARACTERS:
    out = STUDIO / cid / "cutouts"; out.mkdir(exist_ok=True)
    todo = [p for p in sorted((STUDIO / cid / "atlas").glob("*.png"))
            if not (out / p.name).exists() and any(p.stem == j for j, k, _, _ in studio.ATLAS_JOBS if k in ("pose", "view"))]
    if todo:
        for p, im in zip(todo, studio.remove_background([Image.open(p) for p in todo], registry.REVISIONS[studio.SEG_MODEL])):
            im.save(out / p.name)
    log(f"{cid}: {len(todo)} cut-outs")
BENCH["stage_seconds"]["cutout"] = round(time.time() - t0, 1)'''),
    ("source", '''# ==== 8. Atlas source + spec (head boxes, face patches) — REVIEW before importing ====
from PIL import ImageDraw
for cid in STUDIO_CHARACTERS:
    src, spec = studio.build_atlas_source(STUDIO, cid, log)
    review = []
    for pid, P in spec["poses"].items():
        im = Image.open(src / P["file"]).convert("RGBA"); bg = Image.new("RGBA", im.size, (235, 235, 240, 255)); bg.alpha_composite(im)
        d = ImageDraw.Draw(bg)
        if P.get("head"):
            cx, top, chin = P["head"]; d.line([(cx - 90, top), (cx + 90, top)], fill=(0, 0, 255), width=4); d.line([(cx - 90, chin), (cx + 90, chin)], fill=(0, 160, 0), width=4)
        if P.get("face"):
            d.rectangle(P["face"]["box"], outline=(255, 0, 0), width=3)
        review.append((pid, bg))
    studio.contact_sheet(review, STUDIO / f"{cid}_atlas_review.png", cols=6, cell=300)
    print(f"{cid}: check {STUDIO / (cid + '_atlas_review.png')} — fix any wrong box in {src / 'spec.json'}, then run the next cell")'''),
    ("import", '''# ==== 9. Import as a NEW atlas version + previews (then approve with colab/approve_character.py) ====
prov = {"model": studio.FLUX_MODEL, "revision": registry.REVISIONS[studio.FLUX_MODEL], "backend": BACKEND,
        "segmentation": studio.SEG_MODEL, "segmentation_revision": registry.REVISIONS[studio.SEG_MODEL],
        "poc_approval": json.load(open(STUDIO / "poc_approval.json")), "license": "Apache-2.0 model output; project-owned artwork"}
for cid in STUDIO_CHARACTERS:
    src = STUDIO / cid / "atlas_src"
    spec = json.load(open(src / "spec.json"))
    spec["identity"] = studio_run.IDENTITY[cid]
    vdir = ASSETS / "characters" / cid / NEW_VERSION
    if not (vdir / "manifest.json").exists():
        SA.import_atlas(src, ASSETS, cid, NEW_VERSION, spec, prov, log)
    (vdir / "previews").mkdir(exist_ok=True)
    ch = SB.load(cid, NEW_VERSION, ASSETS)
    cells = []
    for g in SB.POSES:
        pid = ch.atlas_pose(g)
        if pid:
            im = Image.new("RGBA", (300, 540), (232, 236, 245, 255))
            SA.AtlasPainter(ch, 0.3).draw(im, (0, 0), pid, {"expr": None})
            cells.append((g, im))
    studio.contact_sheet(cells, vdir / "previews" / "poses.png", cols=7, cell=300)
    cov = SA.coverage(ch.man)
    json.dump(cov, open(vdir / "previews" / "coverage.json", "w"), indent=1)
    print(f"{cid} {NEW_VERSION}: previews in {vdir / 'previews'}; approval blockers: {SB.approval_blockers(vdir)}")
json.dump(BENCH, open(STUDIO / "benchmark.json", "w"), indent=1)
print("To approve (after looking at the previews and a talking test from colab/storybook_previews.py --atlas):")
print(f"  python3 colab/approve_character.py {ASSETS} arjun {NEW_VERSION} --by \\"<your name>\\"")'''),
    ("plates", '''# ==== 10. Optional: generated environment plates (one-time, versioned; same camera presets) ====
import envfactory
if MAKE_ENV_PLATES:
    GEN2 = studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log) if BACKEND == "local" else studio_run.HostedGen(log)
    for (env, set_name), prompt in envfactory.PLATE_PROMPTS.items():
        studio_run.run_job(GEN2, STUDIO / "plates" / f"{env}_{set_name}.png", prompt, [], (1024, 576), 500,
                           {"environment": env, "set": set_name}, log)
    for env in sorted({e for e, _ in envfactory.PLATE_PROMPTS}):
        plates = {s: STUDIO / "plates" / f"{e}_{s}.png" for (e, s) in envfactory.PLATE_PROMPTS if e == env and (STUDIO / "plates" / f"{e}_{s}.png").exists()}
        if plates and not (ASSETS / "environments" / env / "v2" / "manifest.json").exists():
            envfactory.import_plates(ASSETS, env, "v2", plates, provenance={"model": studio.FLUX_MODEL})'''),
]


def main(out):
    cells = [("markdown", MD, None)]
    for tag, src in CELLS:
        if tag == "lib":
            src = build.cell_source("lib", "cast").replace("__LIB_FILES__", json.dumps(build.lib_files(studio=True), indent=0))
        cells.append(("code", src, tag))
    build.write_notebook(cells, out)


if __name__ == "__main__":
    main(sys.argv[1])
