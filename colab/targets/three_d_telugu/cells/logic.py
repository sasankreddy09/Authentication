# ==== 8c. Logical consistency: controlled generation + checks for shots with a scene-state ====
# Shots with a "state" block in the plan are generated with Wan2.1-VACE-1.3B, steered by a layout video drawn from the
# plan (white background, one coloured box per person/object, moving where the plan says). Every clip is then checked
# against the SAME plan: people/objects detected with OWLv2, camera motion measured, positions, direction, stops,
# hand-offs and order checked (engine/lib/motion_check.py), and each required action asked to a video-language model
# (Qwen3-VL-2B). Failures get a TARGETED repair (stronger layout, simpler layout, static camera, new seed, fewer frames)
# within REPAIR_BUDGET; whatever still fails or is uncertain goes to REVIEW and is NOT used unless approved in
# approvals.json — the editor uses the shot's drawn fallback instead. Nothing fails silently.
VACE_MID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"            # Apache-2.0; same text encoder and VAE as Wan2.1-T2V-1.3B
VACE_NATIVE = (16, 81, 832, 480)                          # fps, max frames, width, height
GEN_CODE_VERSION = 1                                      # bump when generation code changes (regenerates)
CHECK_FRAMES = 12                                         # frames sampled for detection and the VLM
STATES = globals().get("STATES", {})
SCENE_ENTS = globals().get("SCENE_ENTS", {})
APPROVALS_PATH = PROJ / "approvals.json"
APPROVALS = json.loads(APPROVALS_PATH.read_text()) if APPROVALS_PATH.exists() else {}
LOGIC_DIR = PROJ / "render" / "review"; LOGIC_DIR.mkdir(parents=True, exist_ok=True)
STATIC_SUFFIX = ", static camera, locked-off tripod shot"
_TRACKER, _JUDGE, _LOADER_ERRORS = None, None, {}
LOCAL_HF = WORK.parent / "hf_checkers"     # checkers (OWLv2 0.6 GB, Qwen3-VL-2B ~4.3 GB) stay on the session disk, not
                                           # on Drive: a free 15 GB Drive is already used by the video model caches


def logic_shot(shot):
    return VIDEO_RECIPE == "auto" and shot["shot_id"] in STATES


def logic_frames(shot):
    fps, most = VACE_NATIVE[:2]
    return frames_for(TL[shot["shot_id"]]["dur"], fps, most) if FRAMES_ON_DEMAND else most


def logic_prompt(shot, static=False):
    return clip_prompt(shot, "t2v") + (STATIC_SUFFIX if static else "")


def _hash(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16]


def logic_gen_key(shot):
    st = STATES[shot["shot_id"]]
    return acceptance.generation_key(
        model=VACE_MID, recipe="vace_layout", steps=TURBO_STEPS if VIDEO_SPEED == "turbo" else CLIP_STEPS,
        lora=TURBO_LORA[1] if VIDEO_SPEED == "turbo" else None, frames=logic_frames(shot), size=VACE_NATIVE[2:],
        seed=shot["seed"], prompt=logic_prompt(shot), negative=NEG_V, control=_hash(st.model_dump()),
        code_version=GEN_CODE_VERSION, extra={"budget": REPAIR_BUDGET})


def logic_val_key(shot):
    st = STATES[shot["shot_id"]]
    return acceptance.validation_key(
        logic_gen_key(shot), {"state": st.model_dump(), "queries": entity_queries(st)},
        {"scene_state": scene_state.VERSION, "motion": motion_check.VERSION, "tracker": trackers.Owlv2Tracker.MODEL,
         "vlm": (vlm_check.VERSION, vlm_check.QwenJudge.MODEL) if VLM_CHECK else None, "logic": LOGIC_CHECK},
        motion_check.TH)


def entity_queries(st):
    """Detector text query per entity: plan scene_entities[*].query, else the character/prop tag, else the name."""
    q = {}
    for k in set(st.start) | {e.actor for e in st.events} | {e.obj for e in st.events if e.obj}:
        ent = (PLAN.get("scene_entities") or {}).get(k, {})
        q[k] = ent.get("query") or (CHARS.get(k, {}) if "CHARS" in globals() else {}).get("tag") or \
            (PLAN.get("props", {}).get(k, {}).get("tag")) or ent.get("name") or k
    return q


def logic_usable(shot):
    sc = read_sidecar(PROJ / "clips" / shot["shot_id"]) or {}
    ok, why = acceptance.usable(dict(sc, shot_id=shot["shot_id"]), logic_gen_key(shot), logic_val_key(shot), APPROVALS)
    if ok and not all((PROJ / "clips" / p).exists() for p in sc.get("parts", [])):
        return False, "clip file missing: regenerate"
    return ok, why


CHECKER_LOADS = {"n": 0, "s": 0.0}


def _load_checkers():
    global _TRACKER, _JUDGE
    dev = "cuda" if GPU["name"] else "cpu"
    if (LOGIC_CHECK and _TRACKER is None and "tracker" not in _LOADER_ERRORS) or \
            (VLM_CHECK and _JUDGE is None and "vlm" not in _LOADER_ERRORS):
        CHECKER_LOADS["n"] += 1
    t0 = time.time()
    if LOGIC_CHECK and _TRACKER is None and "tracker" not in _LOADER_ERRORS:
        try:
            _TRACKER = trackers.Owlv2Tracker(device=dev, cache_dir=str(LOCAL_HF))
            log(f"OWLv2 detector loaded, GPU {gpu_mem()}"); ram("(OWLv2 detector loaded)")
        except Exception:
            log_exc("OWLv2 could not load"); _LOADER_ERRORS["tracker"] = "detector could not load"
    if VLM_CHECK and _JUDGE is None and "vlm" not in _LOADER_ERRORS:
        try:
            _JUDGE = vlm_check.QwenJudge(device=dev, cache_dir=str(LOCAL_HF))
            log(f"Qwen3-VL judge loaded, GPU {gpu_mem()}"); ram("(Qwen3-VL judge loaded)")
        except Exception:
            log_exc("Qwen3-VL could not load"); _LOADER_ERRORS["vlm"] = "video-language model could not load"
    CHECKER_LOADS["s"] += time.time() - t0


def logic_checks(shot, arr, fps):
    """Every check for one clip: list of CheckResult (PASS / FAIL / UNKNOWN, with evidence)."""
    sid, st = shot["shot_id"], STATES[shot["shot_id"]]
    n = len(arr)
    idx = np.linspace(0, n - 1, min(CHECK_FRAMES, n)).round().astype(int)
    exp_all = layout_control.expected_tracks(st, n)
    exp = {k: [v[i] for i in idx] for k, v in exp_all.items()}
    res = []
    if not LOGIC_CHECK:
        return [motion_check.CheckResult("logic_check", "UNKNOWN", None, "LOGIC_CHECK is off: clip not verified")]
    _load_checkers()
    if _TRACKER is None:
        return [motion_check.CheckResult("validator_error", "UNKNOWN", None, _LOADER_ERRORS.get("tracker", "no detector"))]
    obs = _TRACKER.track(arr[idx], entity_queries(st))
    near_obs = [obs_frame for obs_frame in ([b[0] for b in (obs[k][j] for k in obs) if b] for j in range(len(idx)))]
    # camera motion is measured on the BACKGROUND: blank where subjects are detected (nearest sampled frame) AND where
    # the plan puts every person / prop / moving thing in that exact frame (the detector can miss someone; the plan
    # can't). Static scenery (anchors such as a train coach) stays in: it is what shows a real camera move.
    movers = {k for k, v in exp_all.items()
              if getattr(SCENE_ENTS.get(k), "kind", "character") != "anchor" or (v and v[0] != v[-1])}
    masks = [near_obs[int(np.argmin(np.abs(idx - i)))] + [exp_all[k][i] for k in movers if exp_all[k][i]]
             for i in range(n)]
    shift, rel = motion_check.estimate_camera_motion(arr, masks, scale=8)
    src_len = n / fps
    stretch = min(max(TL[sid]["dur"] / src_len, 1.0), MAX_SLOWMO)
    res += motion_check.run_checks(motion_check.validate_clip, st, exp, obs, shift[idx], rel[idx], SCENE_ENTS,
                                   clip_seconds=src_len, used_seconds=TL[sid]["dur"], stretch=stretch)
    # actions boxes can't prove (hands, small objects, scans): ask the VLM, cross-checked with the geometry
    for e in st.events:
        # every required action that isn't pure walking/standing is asked: an unchecked required action would
        # otherwise pass silently (e.g. "the ticket checker looks at her phone")
        if not e.critical or e.type in {"stop", "idle"}:
            continue
        motion = e.type in scene_state.LOCOMOTION
        if motion:
            # walking / entering / exiting is measured by geometry (start, end, direction of the actor). Before,
            # "enter"/"exit" looked for an object hand-off result (never there) → geometry UNKNOWN → the VLM alone
            # decided, and a "no" from a 2B model on 12 frames triggered a blind reseed.
            mine = [r for r in res if r.entity == e.actor and r.check in ("start_position", "end_position", "direction")]
            gst = ("FAIL" if any(r.status == "FAIL" for r in mine) else
                   "PASS" if mine and all(r.status == "PASS" for r in mine) else "UNKNOWN")
            geo = motion_check.CheckResult("geometry", gst, e.actor, "; ".join(f"{r.check} {r.status}" for r in mine))
            if gst != "UNKNOWN":       # geometry measured it (as walk_to always was): no second opinion needed
                res.append(motion_check.CheckResult("event_motion", gst, e.actor, f"{e.id} ({e.type}): geometry {gst}",
                                                    {"event": e.id, "geometry": geo.reason}))
                continue
        else:
            geo = next((r for r in res if r.check == "object_transfer" and r.entity == e.obj), None)
        if not VLM_CHECK:
            vlm_status, ev = "UNKNOWN", {"why": "VLM_CHECK is off"}
        elif _JUDGE is None:
            vlm_status, ev = "UNKNOWN", {"why": _LOADER_ERRORS.get("vlm", "no VLM")}
        else:
            try:
                names = {k: v for k, v in entity_queries(st).items()}
                ans = _JUDGE.ask(arr[idx], [i / fps for i in idx], vlm_check.question(e, names))
                vlm_status, ev = vlm_check.parse(ans)
                ev["question"] = vlm_check.question(e, names)
            except Exception as ex:
                log_exc(f"{sid}: VLM question failed"); vlm_status, ev = "UNKNOWN", {"why": repr(ex)[:200]}
        geo_status = geo.status if geo else "UNKNOWN"
        # geometry is primary; the VLM can confirm what boxes can't see; a disagreement is never a pass
        final = ("FAIL" if geo_status == "FAIL" else
                 "UNKNOWN" if geo_status == "PASS" and vlm_status == "FAIL" else
                 "PASS" if "PASS" in (geo_status, vlm_status) and "FAIL" not in (geo_status, vlm_status) else
                 vlm_status)
        if geo and geo.status == "UNKNOWN" and final == "PASS":
            geo.critical = False                                  # superseded by the VLM's timed evidence
        if final == "PASS" and isinstance(ev.get("end_s"), (int, float)):
            # the VLM's time is the timing evidence when boxes can't time the action (small objects)
            for j, r in enumerate(res):
                if r.check == "completes_before_cut" and r.entity == e.id and r.status == "UNKNOWN":
                    res[j] = motion_check.check_completes_before_cut(e.id, min(ev["end_s"] / src_len, 1.0), src_len,
                                                                     TL[sid]["dur"], stretch)
        res.append(motion_check.CheckResult("event_motion" if motion else "event", final, e.actor,
                                            f"{e.id} ({e.type}): geometry {geo_status}, VLM {vlm_status}",
                                            dict(ev, event=e.id)))
    return res


CLIP_TIMING = {}     # shot_id -> seconds per stage of its latest generation (sidecar, perf.json, benchmark)


def make_vace_clip(pipe_v, shot, emb, params):
    """One VACE generation steered by the plan's layout video. Every stage is timed with the GPU synchronized."""
    import torch
    st = STATES[shot["shot_id"]]
    fps, _, w, h = VACE_NATIVE
    nf = params["frames"]
    t0 = cuda_now()
    tracks = layout_control.expected_tracks(st, nf)
    ctrl = layout_control.control_video(tracks, w, h, skip=params["skip"])
    mask = [Image.new("L", (w, h), 255)] * nf
    seed = shot["seed"] + params["seed_off"]
    torch.manual_seed(seed)
    vt = getattr(getattr(pipe_v, "vae", None), "timing", None)
    v0 = dict(vt) if vt else None
    steps = []
    def _step(pipe, i, t, kw):
        steps.append(cuda_now())
        return {}
    t1 = cuda_now()
    with GpuSampler() as gs:
        out = pipe_v(video=ctrl, mask=mask, conditioning_scale=params["cs"],
                     prompt_embeds=emb[logic_prompt(shot, params["static"])].to("cuda"),
                     negative_prompt_embeds=emb[NEG_V].to("cuda"), height=h, width=w, num_frames=nf,
                     num_inference_steps=pipe_v.speed["steps"], guidance_scale=pipe_v.speed["guidance"],
                     generator=torch.Generator("cpu").manual_seed(seed), output_type="pt",
                     callback_on_step_end=_step).frames[0]
    t2 = cuda_now()
    if not torch.isfinite(out).all():
        raise RuntimeError("the model produced a broken clip (NaN)")
    arr = (out.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    del out
    t3 = cuda_now()
    tm = {"frames": nf, "steps": pipe_v.speed["steps"], "speed": pipe_v.speed["speed"],
          "layout_video_s": round(t1 - t0, 1), "pipeline_s": round(t2 - t1, 1), "to_frames_s": round(t3 - t2, 1)}
    if vt:
        enc, dec = vt["encode_s"] - v0["encode_s"], vt["decode_s"] - v0["decode_s"]
        start = vt["last_encode_end"] or t1
        tm.update(vae_encode_s=round(enc, 1), vae_decode_s=round(dec, 1),
                  vae_encodes=vt["encode_calls"] - v0["encode_calls"], zero_latent_reused=vt["zero_reused"] - v0["zero_reused"],
                  vae_tiles=bool(getattr(pipe_v.vae, "use_tiling", False)))
        if steps:
            per = np.diff([start] + steps)
            tm.update(denoise_s=round(steps[-1] - start, 1), step_s=[round(float(x), 1) for x in per],
                      step_avg_s=round(float(per.mean()), 1))
        tm["other_s"] = round(tm["pipeline_s"] - enc - dec - tm.get("denoise_s", 0.0), 1)
    tm.update(gs.summary())
    CLIP_TIMING[shot["shot_id"]] = tm
    log(f"{shot['shot_id']}: VACE layout clip {nf} frames, conditioning {params['cs']}, seed {seed}, "
        f"skip {sorted(params['skip'])}, static {params['static']} | timing {json.dumps(tm)}")
    return arr, fps


def precheck(shot):
    """Deterministic checks of the PLAN before any GPU time is spent: problems here make a clip unverifiable no
    matter how well it is generated (the detector can't see a tiny subject; a subject that is on screen too briefly
    fails 'presence' by construction). Returns warnings (shown, logged, not blocking)."""
    st = STATES[shot["shot_id"]]
    n = logic_frames(shot)
    tr = layout_control.expected_tracks(st, n)
    out = []
    for e in st.events:
        if not e.critical:
            continue
        for k in {e.actor, e.obj} - {None}:
            boxes = tr.get(k) or []
            vis = [b for b in boxes if b and b[2] > 0.02 and b[0] < 0.98]
            share = len(vis) / max(len(boxes), 1)
            if share < motion_check.TH["visible"] and not getattr(SCENE_ENTS.get(k), "small", False):
                out.append(f"{k} is on screen in only {share:.0%} of the planned clip (presence needs "
                           f"{motion_check.TH['visible']:.0%})")
            if vis and k == e.actor and min((b[2] - b[0]) * (b[3] - b[1]) for b in vis) < 0.01:
                out.append(f"{k}'s box is under 1% of the frame at some point: the detector may miss it")
    return out


def make_vace_clip_safe(pipe_v, shot, emb, params):
    """make_vace_clip, but an out-of-memory error doesn't lose the shot: the memory state is logged, memory is freed
    and the clip is retried once with 25 % fewer frames (attention memory grows with the square of the frame count)."""
    import torch
    try:
        return make_vace_clip(pipe_v, shot, emb, params)
    except torch.cuda.OutOfMemoryError:
        try:
            summary = torch.cuda.memory_summary(abbreviated=True)
        except Exception:
            summary = "(memory summary unavailable)"
        log(f"{shot['shot_id']}: out of GPU memory ({gpu_mem()})\n" + summary, "ERROR")
        free_gpu()
        params = apply_repair("fewer_frames", params, STATES[shot["shot_id"]])
        print(f"    {shot['shot_id']}: GPU memory full → retrying with {params['frames']} frames")
        return make_vace_clip(pipe_v, shot, emb, params)


def release_checkers():
    """Give the checkers' GPU memory back before the video model is loaded again."""
    global _TRACKER, _JUDGE
    for obj, name in ((_TRACKER, "OWLv2"), (_JUDGE, "Qwen3-VL")):
        if obj is not None:
            release_model(getattr(obj, "model", None), name)
    _TRACKER = _JUDGE = None
    free_gpu()


def apply_repair(rep, params, st):
    """Targeted repair: change exactly what the failure points at."""
    if rep == "layout_strength":
        params["cs"] = min(params["cs"] + 0.25, 1.5)
    elif rep == "layout_simplify":                          # keep only moving subjects and what they interact with
        keep = {e.actor for e in st.events} | {e.obj for e in st.events if e.obj} | {e.target for e in st.events if e.target}
        params["skip"] = set(st.start) - keep
    elif rep == "static_camera":
        params["static"] = True
    elif rep == "reseed":
        params["seed_off"] += 1000
    elif rep == "fewer_frames":
        f = int(params["frames"] * 0.75)                       # 25 % shorter, kept at Wan's 4k+1 frame counts
        params["frames"] = max(33, 4 * ((f - 1) // 4) + 1)
    return params


def review_assets(shot, arr, results, decision):
    """Pictures for a person: 4 frames of the clip, and the plan's intended layout (first → last positions)."""
    sid, st = shot["shot_id"], STATES[shot["shot_id"]]
    try:
        n = len(arr)
        strip = np.concatenate([np.asarray(Image.fromarray(arr[i]).resize((416, 240)))
                                for i in np.linspace(0, n - 1, 4).round().astype(int)], axis=1)
        Image.fromarray(strip).save(LOGIC_DIR / f"{sid}.jpg", quality=85)
        names = {k: (CHARS.get(k, {}).get("name") if "CHARS" in globals() else None) or k for k in st.start}
        layout_control.composition_guide(st, layout_control.expected_tracks(st, 9), 832, 480, names).save(
            LOGIC_DIR / f"{sid}_plan.png")
        (LOGIC_DIR / f"{sid}.txt").write_text(
            f"{sid}: {decision.status}\nfailed: " + "; ".join(decision.reasons) + "\nuncertain: " + "; ".join(decision.unknown)
            + "\n\n" + "\n".join(f"{r.status:7} {r.check} {r.entity or ''}: {r.reason}" for r in results))
    except Exception:
        log_exc(f"{sid}: could not save review pictures")


def decode_clip(path):
    """mp4 -> uint8 frames (for re-validating a cached clip without regenerating it)."""
    probe = subprocess.run(["ffmpeg", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r", (\d+)x(\d+)[, ]", probe)
    w, h = int(m.group(1)), int(m.group(2))
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


print(f"Logical consistency: {len(STATES)} shots with a scene-state"
      + (f" → VACE layout control + checks (LOGIC_CHECK={LOGIC_CHECK}, VLM_CHECK={VLM_CHECK}, "
         f"REPAIR_BUDGET={REPAIR_BUDGET})" if STATES else "; other AI shots keep plain text-to-video (not logically verified)"))
