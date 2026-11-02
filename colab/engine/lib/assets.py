"""Licensed real images from Wikimedia Commons, ranked for relevance to what the narration says.

search (Commons API, polite client from research.Wiki) → licence policy (registry.asset_license: PD / CC0 / CC BY;
BY-SA only on opt-in; unknown rejected) → size/type filter → download (cached, sha256) → SigLIP 2 relevance ranking
→ decision with evidence → assets_manifest.json (creator, licence, URLs, hash, query, scores).
A candidate is accepted only if it matches the beat's visual description better than generic distractors AND clears
a calibrated floor; otherwise the beat falls back to a deterministic card. Never "any picture rather than none".
"""
from __future__ import annotations

import hashlib, html, json, re, urllib.request
from pathlib import Path

from registry import asset_license

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
RANK_MODEL = "google/siglip2-base-patch16-224"
TH = {"min_width": 1000, "floor_logit": -6.0, "margin": 1.5,   # calibrated on real Commons images (right +3.9..+6.5,
      "narration_margin": 0.0}                                   # wrong ≤ -2.7): see docs/10 and tests/test_assets.py
DISTRACTORS = ["a page of text", "a logo or icon", "a chart or table of numbers", "a photo of a person's face",
               "a map", "a screenshot of a website", "a building", "an empty white background"]


def _clean(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h or ""))).strip()


def search(wiki, query, n=12, allow_sharealike=False):
    """Commons files for a query, each with its licence decision (rejected ones kept, with the reason)."""
    d = wiki.get(action="query", generator="search", gsrsearch=f"{query} filetype:bitmap", gsrnamespace="6",
                 gsrlimit=str(n), prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth="1600",
                 iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|AttributionRequired|ImageDescription|Credit")
    out = []
    for pg in sorted(d.get("query", {}).get("pages", []), key=lambda p: p.get("index", 0)):
        ii = (pg.get("imageinfo") or [{}])[0]
        md = ii.get("extmetadata", {})
        get = lambda k: (md.get(k) or {}).get("value", "")
        ok, why = asset_license(_clean(get("LicenseShortName")), get("LicenseUrl"), allow_sharealike)
        if ok and ii.get("width", 0) < TH["min_width"]:
            ok, why = False, f"too small ({ii.get('width')} px wide)"
        if ok and ii.get("mime") not in ("image/jpeg", "image/png", "image/svg+xml", "image/tiff", "image/webp"):
            ok, why = False, f"not a picture ({ii.get('mime')})"
        out.append({"id": "C" + hashlib.sha1(pg["title"].encode()).hexdigest()[:10], "title": pg["title"],
                    "page": ii.get("descriptionurl"), "file_url": ii.get("thumburl") or ii.get("url"),
                    "width": ii.get("width"), "height": ii.get("height"), "mime": ii.get("mime"),
                    "license": _clean(get("LicenseShortName")), "license_url": get("LicenseUrl"),
                    "author": _clean(get("Artist"))[:200] or "unknown", "credit": _clean(get("Credit"))[:200],
                    "attribution_required": str(get("AttributionRequired")).lower() in ("true", "1", "yes"),
                    "description": _clean(get("ImageDescription"))[:300], "query": query,
                    "license_ok": ok, "license_why": why})
    return out


def download(asset, folder, ua):
    """Fetch the (1600 px) rendition once; the file name is its content hash (cache + provenance)."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    known = folder / f"{asset['id']}.json"
    if known.exists():
        meta = json.loads(known.read_text())
        if (folder / meta["file"]).exists():
            asset.update(file=meta["file"], sha256=meta["sha256"]); return folder / meta["file"]
    req = urllib.request.Request(asset["file_url"], headers={"User-Agent": ua})
    data = urllib.request.urlopen(req, timeout=60).read()
    sha = hashlib.sha256(data).hexdigest()
    ext = {"image/png": ".png", "image/svg+xml": ".png"}.get(asset["mime"], ".jpg")   # SVGs come as PNG renditions
    f = folder / f"{sha[:16]}{ext}"
    f.write_bytes(data)
    asset.update(file=f.name, sha256=sha)
    known.write_text(json.dumps({"file": f.name, "sha256": sha}))
    return f


class Ranker:
    """SigLIP 2 image–text logits. Used RELATIVELY: the intended description must beat generic distractors by a
    margin and clear a floor. A high score is evidence of relevance, never proof that the picture is correct."""
    def __init__(self, model=RANK_MODEL, device="cpu", cache_dir=None):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.device = torch, device
        self.proc = AutoProcessor.from_pretrained(model, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model, cache_dir=cache_dir).to(device).eval()

    def logits(self, images, texts):
        """[PIL images] × [texts] → matrix of logits (rows: images)."""
        with self.torch.inference_mode():
            inp = self.proc(text=texts, images=images, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            return self.model(**inp).logits_per_image.float().cpu().numpy()


def rank(ranker, files, description, others=(), th=TH, narration=None, other_narration=()):
    """Score candidate images for a beat. Accepted only if the picture fits (1) the beat's visual description AND
    (2) what the narration actually SAYS, each better than generic distractors and the OTHER beats (a picture that
    fits another beat better is not this beat's picture). Returns [(index, logit, margin, accepted, why)] best first."""
    from PIL import Image
    if not files:
        return []
    imgs = [Image.open(f).convert("RGB") for f in files]
    others = [o for o in others if o and o != description]
    texts = [description] + list(DISTRACTORS) + others
    n_t = len(texts)
    if narration:
        texts += [narration] + [o for o in other_narration if o and o != narration]
    L = ranker.logits(imgs, texts)
    out = []
    for i, row in enumerate(L):
        own, best_other = float(row[0]), float(max(row[1:n_t])) if n_t > 1 else -1e9
        margin = own - best_other
        ok = own >= th["floor_logit"] and margin >= th["margin"]
        why = ("matches the beat" if ok else
               f"below the floor ({own:.1f} < {th['floor_logit']})" if own < th["floor_logit"] else
               f"fits '{texts[1 + int(row[1:n_t].argmax())][:40]}' as well or better (margin {margin:.1f})")
        if ok and narration:
            said = float(row[n_t])
            rivals = list(row[1:1 + len(DISTRACTORS)]) + list(row[n_t + 1:])
            n_margin = said - (max(rivals) if rivals else -1e9)
            if n_margin < th["narration_margin"]:
                ok, why = False, f"fits its description but not what is SAID (narration margin {n_margin:.1f})"
        out.append((i, round(own, 2), round(margin, 2), ok, why))
    return sorted(out, key=lambda r: (-r[3], -r[1]))


def credit_line(a):
    """Short on-screen / description credit, as CC BY requires (TASL: title, author, source, licence)."""
    title = re.sub(r"^File:|\.\w+$", "", a["title"])
    return f"{title} — {a['author']} — {a['license']} — Wikimedia Commons"
