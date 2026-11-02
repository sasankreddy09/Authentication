"""Model and asset licence registry: every model the engine can load, with its VERIFIED weight licence, and the
licence policy for third-party images.

Verified against the Hugging Face API / publisher pages on 2026-10-18 (tests/test_registry.py --online re-checks).
status: "ok" = permissive licence (Apache-2.0 / MIT), "review" = usable but with a condition a person should know
about (see note), "blocked" = must not be used. LICENSE_STRICT in the notebook refuses "review" models.
"""
from __future__ import annotations

import re

MODELS = {
    # id: (use, licence of the weights, status, note)
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": ("text-to-video (T4)", "apache-2.0", "ok", ""),
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers": ("layout-controlled video (scene-state shots)", "apache-2.0", "ok", ""),
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": ("video on big GPUs; its UMT5 text encoder is also used on the T4", "apache-2.0", "ok", ""),
    "Kijai/WanVideo_comfy": ("turbo LoRA (Self-Forcing DMD, 4 steps)", "apache-2.0 (upstream gdhe17/Self-Forcing)", "review",
                             "LoRA extracted from the Apache-2.0 Self-Forcing weights; the repackaging repo has no licence "
                             "tag of its own"),
    "google/owlv2-base-patch16-ensemble": ("open-vocabulary detector (scene checks)", "apache-2.0", "ok", ""),
    "Qwen/Qwen3-VL-2B-Instruct": ("video-language judge (scene checks)", "apache-2.0", "ok", ""),
    "openai/clip-vit-base-patch32": ("legacy continuity check (plain text-to-video shots only)", "mit (OpenAI CLIP repo)", "review",
                                     "no licence tag on the Hub; OpenAI's model card lists deployed use as out of scope"),
    "google/siglip2-base-patch16-224": ("image-text relevance ranking (licensed images)", "apache-2.0", "ok", ""),
    "stabilityai/stable-diffusion-xl-base-1.0": ("optional keyframe images", "openrail++", "review",
                                                 "OpenRAIL++: commercial use allowed with use-based restrictions"),
    "Tongyi-MAI/Z-Image-Turbo": ("optional keyframe images (big GPUs)", "apache-2.0", "ok", ""),
    "hexgrad/Kokoro-82M": ("English voices (kokoro-onnx export of these weights)", "apache-2.0", "ok", ""),
    "ai4bharat/indic-parler-tts": ("Telugu voices", "apache-2.0", "ok", "gated: accept the terms, needs a token"),
    # Character / Environment Studio (one-time asset production, never loaded during a routine video render)
    "black-forest-labs/FLUX.2-klein-4B": ("character concepts, reference-conditioned expressions/poses (Studio)",
                                          "apache-2.0", "ok", ""),
    "black-forest-labs/FLUX.2-klein-9B": ("(not used)", "flux-non-commercial-license", "blocked",
                                          "non-commercial weights: refused (use the Apache-2.0 4B model)"),
    "ZhengPeng7/BiRefNet": ("background removal / layer cut-out of approved artwork (Studio)", "mit", "ok", ""),
    "briaai/RMBG-2.0": ("(not used)", "cc-by-nc-4.0 (bria-rmbg-2.0)", "blocked", "non-commercial weights: refused"),
    "briaai/RMBG-1.4": ("(not used)", "bria-rmbg-1.4 (commercial use needs an agreement)", "blocked",
                        "not commercially usable without a Bria agreement: refused"),
    "Systran/faster-whisper-small.en": ("speech check / word timing (English)", "mit", "ok", ""),
    "Systran/faster-whisper-tiny": ("speech check in tests", "mit", "ok", ""),
    "Systran/faster-whisper-tiny.en": ("speech check in tests (English)", "mit", "ok", ""),
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": ("speech check (Telugu)", "mit", "ok",
                                                     "the Hub redirects it to dropbox-dash/faster-whisper-large-v3-turbo"),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("expressive English narration (instruction-controlled delivery)",
                                             "apache-2.0", "ok", "runs in its own environment (qwen-tts pins transformers 4.57.3)"),
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": ("narration without instruction control (tests only)", "apache-2.0", "ok", ""),
    "speechbrain/spkrec-ecapa-voxceleb": ("narrator identity check (speaker embeddings)", "apache-2.0", "ok", ""),
    "Qwen/Qwen3-4B-Instruct-2507": ("topic → script planner (default)", "apache-2.0", "ok", ""),
    "Qwen/Qwen3-1.7B": ("topic → script planner (low memory)", "apache-2.0", "ok", ""),
    "Qwen/Qwen3-0.6B": ("planner in CPU tests", "apache-2.0", "ok", ""),
    "cross-encoder/nli-deberta-v3-base": ("fact check: is a sentence entailed by its source? (default)", "apache-2.0", "ok", ""),
    "cross-encoder/nli-deberta-v3-small": ("fact check, faster/weaker alternative", "apache-2.0", "ok", ""),
    "ResembleAI/chatterbox-turbo": ("evaluated TTS candidate, NOT integrated", "mit", "review",
                                    "weights MIT; chatterbox-tts pins torch==2.6.0 + transformers==5.2.0 (own env, ~2.5 GB); "
                                    "distinct voices need reference audio (only synthetic/authorised); outputs carry a "
                                    "Perth watermark; not benchmarked here"),
    "ResembleAI/chatterbox-nano": ("evaluated TTS candidate, NOT integrated", "mit", "review",
                                   "same package and conditions as chatterbox-turbo; not benchmarked here"),
    "gdhe17/Self-Forcing": ("upstream of the turbo LoRA (checkpoints/self_forcing_dmd.pt); not loaded yet",
                            "apache-2.0", "ok", "a licence-clean turbo needs these weights converted to a diffusers "
                            "LoRA locally (not done; the Kijai repackaging stays 'review')"),
}
# Hub commit of every model at the time its licence was verified (2026-10-18). Recorded so a later change of the
# weights or of the licence on the Hub is visible (tests/test_registry.py --online reports a moved revision).
# Not yet passed as revision= at every load site.
REVISIONS = {
    "Systran/faster-whisper-tiny.en": "0d3d19a32d3338f10357c0889762bd8d64bbdeba",
    "ResembleAI/chatterbox-turbo": "749d1c1a46eb10492095d68fbcf55691ccf137cd",
    "ResembleAI/chatterbox-nano": "71ccd1d0081b430592cea481f4307e764e07bc64",
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "0fad780a534b6463e45facd96134c9f345acfa5b",
    "Wan-AI/Wan2.1-VACE-1.3B-diffusers": "ec4d2cb062b548996b179d493fdd05340de702a1",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "b8fff7315c768468a5333511427288870b2e9635",
    "Kijai/WanVideo_comfy": "8260d429d19fd7a72304cad059160b95d843913f",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
    "Qwen/Qwen3-VL-2B-Instruct": "89644892e4d85e24eaac8bacfd4f463576704203",
    "openai/clip-vit-base-patch32": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    "google/siglip2-base-patch16-224": "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
    "stabilityai/stable-diffusion-xl-base-1.0": "462165984030d82259a11f4367a4eed129e94a7b",
    "Tongyi-MAI/Z-Image-Turbo": "f332072aa78be7aecdf3ee76d5c247082da564a6",
    "hexgrad/Kokoro-82M": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "ai4bharat/indic-parler-tts": "660bc53c64ae83150613264111708a0465211237",
    "black-forest-labs/FLUX.2-klein-4B": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
    "black-forest-labs/FLUX.2-klein-9B": "92196c8e11f7b6cf2b7493e037d8c5345c559216",
    "ZhengPeng7/BiRefNet": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    "briaai/RMBG-2.0": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
    "briaai/RMBG-1.4": "2ceba5a5efaec153162aedea169f76caf9b46cf8",
    "Systran/faster-whisper-small.en": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
    "Systran/faster-whisper-tiny": "d90ca5fe260221311c53c58e660288d3deb8d356",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo": "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": "0c0e3051f131929182e2c023b9537f8b1c68adfe",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": "85e237c12c027371202489a0ec509ded67b5e4b5",
    "speechbrain/spkrec-ecapa-voxceleb": "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "cross-encoder/nli-deberta-v3-base": "6c749ce3425cd33b46d187e45b92bbf96ee12ec7",
    "cross-encoder/nli-deberta-v3-small": "fa2804872c3b4bd748f38c0185cc85775361e735",
    "gdhe17/Self-Forcing": "2f8b779212da279d212c22a509b66ad6552f350e",
}
# Non-model assets and their licences (media policy: PD / CC0 / CC BY; BY-SA opt-in; NC/ND/unknown rejected)
MEDIA_SOURCES = {
    "characters": ("Arjun and Meera: original artwork drawn by code (engine/lib/charanim.py) from the Character Bible",
                   "original work of this project"),
    "demonstrations": ("finger test, eye views, projector + glasses: drawn by code (engine/lib/demos.py)",
                       "original work of this project"),
    "fonts": ("DejaVu Sans (Bitstream Vera licence) / Hind Guntur (SIL OFL 1.1) for Telugu", "permissive font licences"),
    "music": ("Kevin MacLeod, incompetech.com tracks (engine/lib/musiclib.py), credited in the description", "CC BY 4.0"),
    "sound effects": ("synthesised in the notebook (sfx cell)", "original work of this project"),
    "photos": ("Wikimedia Commons, per image: licence checked by asset_license(), credit drawn on screen + description",
               "PD / CC0 / CC BY (BY-SA only with ALLOW_SHAREALIKE)"),
}
ALLOWED_MODEL_LICENSES = ("apache-2.0", "mit", "bsd")

# The ONE approved hosted exception (2026-10-19): Google Gemini free-tier TTS, for SPEECH SYNTHESIS ONLY. It is not
# open-source and not an open-weight model; it is listed apart from MODELS so it can never pass as one.
HOSTED_EXCEPTIONS = {
    "gemini-3.8-flash-tts": {"provider": "Google AI Studio / Gemini API", "scope": "tts", "tier": "free only",
                             "kind": "proprietary hosted free-tier TTS exception",
                             "data_use": "free-tier input/output used to improve Google products; human review possible",
                             "docs": "https://ai.google.dev/gemini-api/docs/speech-generation",
                             "verified": "2026-10-19: Telugu + 2 speakers; Standard tier free of charge"},
    "gemini-3.8-flash-lite-tts": {"provider": "Google AI Studio / Gemini API", "scope": "tts", "tier": "free only",
                                  "kind": "proprietary hosted free-tier TTS exception",
                                  "data_use": "free-tier input/output used to improve Google products; human review possible",
                                  "docs": "https://ai.google.dev/gemini-api/docs/speech-generation",
                                  "verified": "2026-10-19: Telugu + 2 speakers; Standard tier free of charge"},
}


def hosted_ok(model: str, use: str) -> tuple[bool, str]:
    """A hosted model is allowed only if it is a listed exception AND used for its approved scope (speech only)."""
    h = HOSTED_EXCEPTIONS.get(model)
    if h is None:
        return False, f"{model}: not an approved hosted exception (no hosted API may be used for {use})"
    if use != h["scope"]:
        return False, f"{model}: approved for {h['scope']} only, not for {use}"
    return True, f"{model}: {h['kind']} ({h['tier']})"


# services that must never be required (paid or closed); the engine must not reference them
FORBIDDEN_SERVICES = ["api.openai.com", "api.anthropic.com", "api.elevenlabs.io", "api.runwayml.com",
                      "api.replicate.com", "modal.com", "runpod.io", "api.pexels.com", "pixabay.com/api"]
FORBIDDEN_IMPORTS = ["openai", "anthropic", "elevenlabs", "runwayml", "replicate", "modal", "runpod"]


def model_ok(mid: str, strict: bool = False) -> tuple[bool, str]:
    m = MODELS.get(mid)
    if m is None:
        return False, f"{mid}: not in the licence registry (add it with its verified weight licence)"
    use, lic, status, note = m
    if status == "blocked" or (strict and status == "review"):
        return False, f"{mid}: {lic} — {note or status}"
    return True, f"{mid}: {lic}" + (f" (review: {note})" if status == "review" else "")


# ------------------------------------------------------------------ images (Wikimedia Commons extmetadata)
def asset_license(short_name: str, license_url: str = "", allow_sharealike: bool = False) -> tuple[bool, str]:
    """Decide from Commons' LicenseShortName (free text, so normalised conservatively). Unknown = rejected.
    Allowed by default: public domain, CC0, CC BY (any version). CC BY-SA only when allow_sharealike (an image the
    video builds on can make the whole video a ShareAlike adaptation). NC / ND / GFDL-only / fair use: never."""
    s = re.sub(r"[\s_]+", " ", (short_name or "").strip().lower())
    u = (license_url or "").lower()
    if not s:
        return False, "no licence stated"
    if any(x in s for x in ("fair use", "non-free", "copyrighted", "all rights reserved")):
        return False, f"not free: {short_name}"
    if re.search(r"\bnc\b|noncommercial|non-commercial", s) or "-nc" in u:
        return False, f"non-commercial: {short_name}"
    if re.search(r"\bnd\b|noderiv", s) or "-nd" in u:
        return False, f"no derivatives: {short_name}"
    if s in ("public domain", "pd", "cc0", "cc0 1.0", "cc-zero", "no restrictions") or s.startswith(("pd-", "public domain")) \
            or "publicdomain" in u:
        return True, "public domain / CC0"
    if re.fullmatch(r"cc[ -]by[ -]?(\d\.\d)?( [a-z]+)?", s):
        return True, f"attribution: {short_name}"
    if re.fullmatch(r"cc[ -]by[ -]sa[ -]?(\d\.\d)?( [a-z]+)?", s):
        return (allow_sharealike, f"share-alike: {short_name}" + ("" if allow_sharealike else " (ALLOW_SHAREALIKE is off)"))
    if s.startswith("gfdl"):
        return False, f"GFDL (impractical in video): {short_name}"
    return False, f"unknown licence (rejected): {short_name}"


ACCESS_HELP = ("Indic Parler-TTS is gated (free, Apache-2.0): 1) log in to huggingface.co with the account of your "
               "HF_TOKEN, 2) open https://huggingface.co/ai4bharat/indic-parler-tts and accept the terms "
               "('Agree and access repository'), 3) make sure the token is a READ token in Colab Secrets as HF_TOKEN "
               "with notebook access ON, 4) run the cell again. The pipeline never bypasses the gate.")


def gated_access(repo: str, token: str | None, http=None, revision: str | None = None) -> dict:
    """Can this token download this gated model? (a HEAD request for config.json; nothing is downloaded or
    logged — the token is only sent to huggingface.co). → {"status": PASS | TOKEN_MISSING | TOKEN_INVALID |
    ACCESS_REQUIRED | UNKNOWN, "why", "help"}. UNKNOWN (offline / unexpected answer) is never treated as PASS."""
    if not token:
        return {"status": "TOKEN_MISSING", "why": "no HF_TOKEN secret", "help": ACCESS_HELP}
    if http is None:
        import requests as http
    rev = revision or REVISIONS.get(repo) or "main"
    url = f"https://huggingface.co/{repo}/resolve/{rev}/config.json"
    try:
        r = http.head(url, headers={"Authorization": f"Bearer {token}"}, allow_redirects=False, timeout=20)
        code = r.status_code
    except Exception as e:                                  # offline: say so, don't guess
        return {"status": "UNKNOWN", "why": f"could not reach huggingface.co ({type(e).__name__})", "help": ACCESS_HELP}
    if code in (200, 302, 307):
        return {"status": "PASS", "why": f"HTTP {code}", "help": ""}
    if code == 401:
        return {"status": "TOKEN_INVALID", "why": "HTTP 401: the token is not valid (new READ token?)", "help": ACCESS_HELP}
    if code == 403:
        return {"status": "ACCESS_REQUIRED", "why": "HTTP 403: this account has not accepted the model's terms",
                "help": ACCESS_HELP}
    return {"status": "UNKNOWN", "why": f"HTTP {code}", "help": ACCESS_HELP}


# Tools (not model weights) the storybook system evaluated or can use, with their licences.
TOOLS = {
    "godot": ("2D cutout rig renderer (evaluated, not used: needs a display + OpenGL in Colab)", "mit"),
    "rhubarb": ("optional mouth-shape timing (English recogniser; phonetic mode for other languages)", "mit"),
    "pillow": ("storybook rig compositing (used)", "mit-cmu (HPND)"),
    "ffmpeg": ("video assembly (used)", "lgpl-2.1+/gpl (binary build)"),
}
