"""Trackers: turn sampled frames into observed boxes per entity, for motion_check.py.

All trackers return {entity: [(box, score) or None per frame]} with boxes normalised to 0..1.
Association never looks at the PLANNED positions (that would bias the check towards passing): the first frame
takes the best-scoring detection, later frames take the detection that continues the previous box.

- ColorTracker: exact colour blobs. Deterministic; used by the synthetic regression tests.
- Owlv2Tracker: open-vocabulary detector (google/owlv2-base-patch16-ensemble, Apache-2.0, ~0.6 GB) queried with each
  entity's short description. Runs on the T4 after the video model is unloaded.
"""
from __future__ import annotations

import numpy as np


def _link(cands_per_frame):
    """cands_per_frame: list of [(box, score)] -> one (box, score) or None per frame, by continuity."""
    out, prev = [], None
    for cands in cands_per_frame:
        if not cands:
            out.append(None)
            continue
        if prev is None:
            pick = max(cands, key=lambda c: c[1])
        else:
            pc = ((prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2)
            pick = min(cands, key=lambda c: abs((c[0][0] + c[0][2]) / 2 - pc[0]) + abs((c[0][1] + c[0][3]) / 2 - pc[1])
                       - 0.05 * c[1])
        out.append(pick)
        prev = pick[0]
    return out


class ColorTracker:
    """Finds pixels of an exact RGB colour per entity (synthetic test videos)."""
    def __init__(self, colors: dict[str, tuple], tol: int = 10):
        self.colors, self.tol = colors, tol

    def track(self, frames: np.ndarray, queries=None):
        n, h, w = frames.shape[:3]
        out = {}
        for name, col in self.colors.items():
            seq = []
            for f in frames:
                m = (np.abs(f.astype(int) - np.array(col)).sum(axis=2) <= self.tol)
                if m.sum() < 4:
                    seq.append([])
                    continue
                ys, xs = np.nonzero(m)
                seq.append([((xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h), 1.0)])
            out[name] = _link(seq)
        return out


def load_on(cls, mid, device, **kw):
    """from_pretrained straight onto the GPU (weights never sit in system RAM: the free Colab tier has ~12.7 GB);
    on the CPU, or if that fails, one weight at a time through RAM (low_cpu_mem_usage) and then moved."""
    if device == "cuda":
        try:
            return cls.from_pretrained(mid, device_map={"": 0}, **kw).eval()
        except Exception:
            pass
    return cls.from_pretrained(mid, low_cpu_mem_usage=True, **kw).to(device).eval()


class Owlv2Tracker:
    """Open-vocabulary detection on sampled frames, linked over time by continuity."""
    MODEL = "google/owlv2-base-patch16-ensemble"

    def __init__(self, device="cuda", cache_dir=None, threshold=0.12):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch, self.device, self.th = torch, device, threshold
        self.dt = torch.float16 if device == "cuda" else torch.float32
        self.proc = Owlv2Processor.from_pretrained(self.MODEL, cache_dir=cache_dir)
        self.model = load_on(Owlv2ForObjectDetection, self.MODEL, device, torch_dtype=self.dt, cache_dir=cache_dir)

    def track(self, frames: np.ndarray, queries: dict[str, str]):
        """queries: {entity: "a young woman in a teal hoodie"}. Returns {entity: [(box, score) or None]}."""
        from PIL import Image
        torch = self.torch
        names = list(queries)
        per = {k: [] for k in names}
        for f in frames:
            img = Image.fromarray(f)
            inp = self.proc(text=[[queries[k] for k in names]], images=img, return_tensors="pt")
            inp = {k: (v.to(self.device, self.dt) if v.dtype.is_floating_point else v.to(self.device))
                   for k, v in inp.items()}
            with torch.no_grad():
                out = self.model(**inp)
            side = max(img.size)                    # OWLv2 pads to a square: boxes are relative to that square
            post = getattr(self.proc, "post_process_grounded_object_detection", None) or \
                self.proc.post_process_object_detection           # renamed in transformers 5.x
            res = post(out, threshold=self.th, target_sizes=torch.tensor([[side, side]]))[0]
            w, h = img.size
            for k in names:
                per[k].append([])
            for box, score, lab in zip(res["boxes"].float().cpu().numpy(), res["scores"].float().cpu().numpy(),
                                       res["labels"].cpu().numpy()):
                x0, y0, x1, y1 = box
                per[names[int(lab)]][-1].append(((max(x0, 0) / w, max(y0, 0) / h, min(x1, w) / w, min(y1, h) / h),
                                                 float(score)))
        return {k: _link(v) for k, v in per.items()}
