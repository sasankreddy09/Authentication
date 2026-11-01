"""Character Studio proof of concept from the command line (claude.md Part 5) — no local GPU needed.

Usage:
  python colab/studio_poc.py <out dir> [--backend hosted|local] [--only arjun,meera]
                             [--steps portrait,full_body_master,expression_surprise,view_three_quarter]
                             [--style-ref arjun=path.png --style-ref meera=path.png]
  python colab/studio_poc.py <studio dir> --atlas arjun [--backend local]   (after the design is approved)

hosted: the official Space black-forest-labs/FLUX.2-klein-4B (set HF_TOKEN in your environment for your account's
free quota; anonymous use allowed 3 images on 2026-10-20). local: your GPU (Colab T4). Re-runs skip finished images.
Everything is logged to <out dir>/studio.log. Nothing is approved here.
"""
import argparse, logging, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine" / "lib"))
import registry, studio, studio_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--backend", default="hosted", choices=["hosted", "local"])
    ap.add_argument("--only", default="arjun,meera")
    ap.add_argument("--steps", default=",".join(j[0] for j in studio_run.POC_JOBS))
    ap.add_argument("--style-ref", action="append", default=[], help="cid=path: an art-direction reference image")
    ap.add_argument("--atlas", help="character id: run the pose-atlas jobs from its approved master")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(out / "studio.log"), level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = lambda m: (print(m, flush=True), logging.info(m))
    ok, why = registry.model_ok(studio.FLUX_MODEL, strict=True)
    log(("licence OK: " if ok else "licence REFUSED: ") + why)
    if not ok:
        sys.exit(1)
    gen = studio_run.HostedGen(log) if a.backend == "hosted" else studio_run.LocalGen(registry.REVISIONS[studio.FLUX_MODEL], log)
    log(f"backend: {gen.name} ({gen.revision})")
    refs = {}
    for r in a.style_ref:
        cid, path = r.split("=", 1)
        refs.setdefault(cid, []).append(path)
    try:
        if a.atlas:
            st = studio_run.run_atlas(gen, out, a.atlas, log=log)
        else:
            st = studio_run.run_poc(gen, out, a.only.split(","), a.steps.split(","), refs, log)
    except studio_run.QuotaExhausted as e:
        log(f"GPU quota exhausted ({e}). Finished images are kept; run the same command later (or with HF_TOKEN) to continue.")
        st = {}
    log(f"status: {st}")
    from PIL import Image
    items = [(f"{p.parent.name} {p.stem}", Image.open(p)) for p in sorted(out.glob("*/*.png"))
             if ".partial" not in p.name and "cutout" not in p.name]
    if items:
        studio.contact_sheet(items, out / "contact_sheet.png", cols=4, cell=420)
        log(f"contact sheet: {out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
