"""Acceptance, targeted repair and cache identities for generated clips.

decide(results) -> Decision
  ACCEPT  every critical check PASS (non-critical FAIL/UNKNOWN are reported, never hidden)
  REPAIR  a critical FAIL with a known targeted repair left in the shot's budget
  REVIEW  a critical UNKNOWN, or failures with no repair left: a person must approve (approvals.json) before the
          clip may be used; otherwise the editor uses the shot's deterministic fallback
Nothing is chosen "because it scored highest": a candidate is usable only if ACCEPTED or approved.

Cache identities (separate on purpose):
  generation key: everything that shapes the pixels (model, recipe, steps, LoRA, frames, size, seed, prompt, control
                  video, references, generator code version). Change → regenerate.
  validation key: generation key + the constraints the clip is judged against (scene-state, expected tracks) +
                  validator versions and thresholds. Change → re-validate the existing clip (no regeneration);
                  until then its status is UNKNOWN, not "accepted".
Pure Python; unit-tested in tests/test_acceptance.py.
"""
from __future__ import annotations

import hashlib, json
from dataclasses import dataclass, field

# failure (check name) -> repairs to try, in order. Each repair is something the executor can actually do.
REPAIRS = {
    "direction":          ["layout_strength", "layout_simplify", "fallback"],
    "start_position":     ["layout_strength", "first_frame", "fallback"],
    "end_position":       ["layout_strength", "layout_simplify", "fallback"],
    "stationary":         ["layout_strength", "static_camera", "fallback"],
    "camera_static":      ["static_camera", "layout_strength", "fallback"],
    "teleport":           ["reseed", "fewer_frames", "fallback"],
    "presence":           ["reference_identity", "reseed", "fallback"],
    "stop_before_action": ["split_action", "fallback"],
    "object_transfer":    ["split_action", "layout_strength", "fallback"],
    "event_order":        ["split_action", "fallback"],
    "event":              ["split_action", "reseed", "fallback"],     # VLM: the action didn't happen
    # walking/entering didn't happen as planned (geometry FAIL, or not measurable + VLM "no"): steer the layout
    # harder / simplify it — the layout boxes ARE what drives the walk; a new seed alone rarely fixes a structure
    "event_motion":       ["layout_strength", "layout_simplify", "fallback"],
    "completes_before_cut": ["earlier_action", "fallback"],
    "identity":           ["reference_identity", "fallback"],
}
# what each repair changes (documented; the executor implements the ones marked automatic)
REPAIR_EFFECT = {
    "layout_strength":    "automatic: VACE conditioning_scale +0.25 (max 1.5) and the layout boxes drawn thicker",
    "layout_simplify":    "automatic: drop non-critical subjects from the layout; keep only the moving subject and its target",
    "static_camera":      "automatic: remove camera-move words from the prompt and add 'static camera, locked-off shot'",
    "reseed":             "automatic: new seed, same controls (only for random glitches such as a teleport frame)",
    "fewer_frames":       "automatic: generate fewer frames (shorter motion is more stable), edit stretches gently",
    "first_frame":        "needs a composition keyframe (L4+): condition the first frame on it",
    "reference_identity": "needs a character reference image (VACE reference_images) — L4+ or a prepared reference",
    "split_action":       "plan change: split the shot into shorter shots with one action each → REVIEW",
    "earlier_action":     "plan change: end the action earlier (event t) → REVIEW",
    "fallback":           "use the shot's deterministic fallback (drawn scene / still + move) and flag it",
}
AUTOMATIC = {"layout_strength", "layout_simplify", "static_camera", "reseed", "fewer_frames"}


@dataclass
class Decision:
    status: str                                   # ACCEPT | REPAIR | REVIEW
    repair: str | None = None
    reasons: list = field(default_factory=list)
    unknown: list = field(default_factory=list)


def decide(results, tried: list[str], budget: int = 2, available=frozenset(AUTOMATIC)):
    """results: list of CheckResult (or dicts). tried: repairs already applied to this shot."""
    rs = [r if isinstance(r, dict) else r.to_dict() for r in results]
    crit = [r for r in rs if r.get("critical", True)]
    fails = [r for r in crit if r["status"] == "FAIL"]
    unknown = [r for r in crit if r["status"] == "UNKNOWN"]
    reasons = [f"{r['check']}({r.get('entity') or '-'}): {r['reason']}" for r in fails]
    unk = [f"{r['check']}({r.get('entity') or '-'}): {r['reason']}" for r in unknown]
    if not fails and not unknown:
        return Decision("ACCEPT", reasons=[], unknown=[])
    if fails and len(tried) < budget:
        for r in fails:
            for rep in REPAIRS.get(r["check"], ["fallback"]):
                if rep in available and rep not in tried:
                    return Decision("REPAIR", rep, reasons, unk)
    return Decision("REVIEW", None, reasons, unk)


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def generation_key(*, model: str, recipe: str, steps: int, lora: str | None, frames: int, size: tuple, seed: int,
                   prompt: str, negative: str, control: str | None = None, references: list | None = None,
                   code_version: int = 1, extra: dict | None = None) -> str:
    return _h({"model": model, "recipe": recipe, "steps": steps, "lora": lora, "frames": frames, "size": list(size),
               "seed": seed, "prompt": prompt, "negative": negative, "control": control,
               "references": references or [], "code": code_version, "extra": extra or {}})


def validation_key(gen_key: str, constraints: dict, validator_versions: dict, thresholds: dict) -> str:
    return _h({"gen": gen_key, "constraints": constraints, "validators": validator_versions, "thresholds": thresholds})


def usable(sidecar: dict, gen_key: str, val_key: str, approvals: dict) -> tuple[bool, str]:
    """May the editor use this clip? (True/False, why)."""
    if not sidecar or sidecar.get("gen_key") != gen_key:
        return False, "missing or made with different settings: regenerate"
    if sidecar.get("val_key") != val_key:
        return False, "constraints or validators changed since it was checked: re-validate"
    st = sidecar.get("decision")
    if st == "ACCEPT":
        return True, "accepted by the checks"
    ap = approvals.get(sidecar.get("shot_id", ""), {})
    if ap.get("approved") and ap.get("gen_key") == gen_key:
        return True, f"approved by a person ({ap.get('note', '')})"
    return False, f"{st or 'unchecked'}: needs approval in approvals.json, otherwise the fallback is used"
