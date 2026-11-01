"""Approve a character (or environment) asset version after a person has looked at its previews.

Usage:  python3 colab/approve_character.py <asset root> <character id> <version> --by "<name>" [--note "..."]

Only status changes (provisional → approved) with who/when; the layers can never change afterwards. Prints the
manifest sha256 that plans pin (build_story_plan.py --visual storybook_v1 picks it up automatically).
"""
import argparse, logging, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine" / "lib"))
import storybook as SB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("character"); ap.add_argument("version")
ap.add_argument("--by", required=True, help="who looked at the previews and approves this artwork")
ap.add_argument("--note", default="")
a = ap.parse_args()
logging.basicConfig(filename=str(Path(a.root) / "approvals.log"), level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
vdir = Path(a.root) / "characters" / a.character / a.version
probs = SB.approval_blockers(vdir)          # files verify, previews exist, licence OK, atlas coverage complete
if probs:
    logging.error("refused %s: %s", vdir, probs)
    sys.exit("refused — fix these first:\n  " + "\n  ".join(probs))
sha = SB.approve(vdir, a.by, a.note)
logging.info("approved %s by %s (%s): manifest %s", vdir, a.by, a.note, sha)
print(f"approved {a.character} {a.version} by {a.by}: manifest sha256 {sha}")
print("next: copy this version folder into colab/assets/characters/ and rebuild the plan with "
      f"build_story_plan.py ... --visual storybook_v1 --char-version {a.version}")
