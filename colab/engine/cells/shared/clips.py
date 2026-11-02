# ==== 9. AI video for every shot (Wan 2.2 on L4/A100, Wan 2.1 small model on T4) ====
NATIVE = {"wan22-5b": (24, 121, 1280, 704), "wan21-1.3b": (16, 81, 832, 480)}   # fps, frames, width, height
TL = {r["shot_id"]: r for r in TIMELINE}

def clip_key(shot, mode):
    k = {"motion_prompt": shot["motion_prompt"], "image_prompt": shot["image_prompt"], "seed": shot["seed"]}
    if mode == "i2v":                                 # only image-to-video clips depend on the keyframe
        k["keyframe_prompt"] = (read_sidecar(PROJ / "keyframes" / f"{shot['shot_id']}.png") or {}).get("prompt")
    return k

def clip_info(shot):
    """The finished clip set for this shot, or None if missing / made from different settings.
    Scene-state shots: only when generated from the same request AND accepted by the checks (or approved)."""
    if logic_shot(shot):
        return read_sidecar(PROJ / "clips" / shot["shot_id"]) if logic_usable(shot)[0] else None
    sc = read_sidecar(PROJ / "clips" / shot["shot_id"])
    if not sc or any(sc.get(k) != v for k, v in clip_key(shot, sc.get("mode")).items()):
        return None
    if sc.get("writer", 1) < CLIP_WRITER:            # made by the old writer: colours are inverted, remake it
        return None
    return sc if all((PROJ / "clips" / p).exists() for p in sc["parts"]) else None

def encode_prompts(texts):
    """Encoded prompts come from the Drive cache when possible. The rest go through Wan's huge text encoder
    (UMT5-XXL), loaded straight onto the GPU once, then freed. Keeping it in system RAM crashed the free T4."""
    import torch
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in texts:
        if emb_path(t).exists():
            try:
                e = torch.load(emb_path(t)).float()
                out[t] = torch.cat([e, e.new_zeros(512 - len(e), e.shape[1])]).unsqueeze(0)   # same as diffusers
            except Exception:
                log_exc(f"cached prompt {emb_path(t).name} unreadable, encoding again")
    missing = [t for t in texts if t not in out]
    print(f"Prompts: {len(out)} from the Drive cache, {len(missing)} to encode.")
    if not missing:
        return out
    from transformers import AutoTokenizer, UMT5EncoderModel
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    need_disk(13, "the text encoder download (11.4 GB)", purge=[([VIDEO_MID["wan21-1.3b"]], ["transformer", "vae"])])
    print("Downloading the text encoder (11 GB, one time per session; takes ~5–10 min)...")
    for mid in TEXT_ENCODER_MIDS:
        try:
            tok = AutoTokenizer.from_pretrained(mid, subfolder="tokenizer")
            for dt in ([torch.bfloat16] if real_bf16() else [torch.float16, torch.bfloat16]):
                te = UMT5EncoderModel.from_pretrained(mid, subfolder="text_encoder", torch_dtype=dt,
                                                      device_map={"": 0}, low_cpu_mem_usage=True).eval()
                ram("(text encoder on GPU)")
                new, ok = {}, True
                with torch.no_grad():
                    for t in missing:
                        ti = tok([prompt_clean(t)], padding="max_length", max_length=512, truncation=True,
                                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
                        n = int(ti.attention_mask.sum())
                        e = te(ti.input_ids.to("cuda"), ti.attention_mask.to("cuda")).last_hidden_state[0, :n].float().cpu()
                        if not torch.isfinite(e).all():
                            ok = False; break
                        new[t] = e
                release_model(te, "text encoder"); del te       # really free its ~11 GB before the video model
                if ok:
                    for t, e in new.items():
                        torch.save(e.half(), emb_path(t))
                        out[t] = torch.cat([e, e.new_zeros(512 - len(e), e.shape[1])]).unsqueeze(0)
                    # the prompts are saved on Drive now, so the 11 GB text encoder can leave the Colab disk
                    print(f"Prompts saved; freed {purge_hf(TEXT_ENCODER_MIDS, ['text_encoder']):.1f} GB of disk.")
                    return out
                print(f"  text encoder overflowed in {dt}, retrying")
        except Exception as e:
            log_exc(f"text encoder from {mid} failed")
            print(f"  text encoder from {mid} failed ({repr(e)[:200]}), trying the next copy")
            free_gpu()
    raise RuntimeError("text encoder could not encode the prompts")

# Speed-up for the T4 model (Wan 2.1 1.3B): the Self-Forcing DMD distillation as a LoRA (Apache-2.0,
# github.com/guandeh17/Self-Forcing; LoRA extracted by Kijai). 4 steps and no negative-prompt pass instead of
# 25 steps x 2 passes: ~10x less work per clip.
TURBO_LORA = ("Kijai/WanVideo_comfy", "LoRAs/Wan2_1_self_forcing_1_3B/Wan2_1_self_forcing_dmd_1_3B_lora_rank_32_fp16.safetensors")
TURBO_STEPS = 4

def turbo_scheduler():
    """Self-Forcing's own sampling: timesteps 1000/750/500/250 warped by shift 5; at each step predict the clean
    video, then re-noise to the next level (stochastic_sampling)."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    class TurboScheduler(FlowMatchEulerDiscreteScheduler):
        def set_timesteps(self, num_inference_steps=None, device=None, **kw):
            n = num_inference_steps
            return super().set_timesteps(sigmas=list(1.0 - np.arange(n) / n), device=device)
    return TurboScheduler(shift=5.0, stochastic_sampling=True)

def turbo_lora_file():
    """The turbo LoRA as a plain local file. Downloaded to the session disk (Hugging Face's cache uses symlinks and
    lock files, which a Google Drive folder doesn't reliably support), then kept on Drive as ONE plain file so the
    next session copies it instead of downloading. Returns the local path."""
    from huggingface_hub import hf_hub_download
    name = Path(TURBO_LORA[1]).name
    keep, local = MODEL_CACHE / "lora" / name, WORK.parent / "lora" / name
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    if keep.exists() and keep.stat().st_size > 50e6:
        shutil.copy(keep, local)
        return local
    got = Path(hf_hub_download(TURBO_LORA[0], TURBO_LORA[1], cache_dir=str(WORK.parent / "hf_lora")))
    shutil.copy(got, local)
    try:
        keep.parent.mkdir(parents=True, exist_ok=True); shutil.copy(local, keep)
    except Exception:
        log_exc("could not keep the turbo LoRA on Drive (it is downloaded again next session)")
    return local

def fuse_turbo(p):
    """Fuse the Self-Forcing turbo LoRA into the transformer and PROVE it worked: the base blocks must change and
    (for VACE) the layout-control blocks must not. Wan2.1-VACE-1.3B's base blocks are bit-identical to Wan2.1-T2V-1.3B
    (checked: 4 sampled tensors, relative difference 0), so this T2V-distilled LoRA lands on the network it was
    distilled from; VACE's own blocks are untouched. Raises with the failing stage named."""
    import torch
    from safetensors.torch import load_file
    stage, t0 = "download", time.time()
    try:
        path = turbo_lora_file()
        stage = "read"; sd = load_file(str(path))
        tr = p.transformer
        base = tr.blocks[0].attn1.to_q.weight
        vace = getattr(tr, "vace_blocks", None)
        vace_w = vace[0].proj_out.weight if vace is not None else None
        before = (base.detach().float().sum().item(), None if vace_w is None else vace_w.detach().float().sum().item())
        stage = "load"; p.load_lora_weights(sd, adapter_name="turbo")
        stage = "fuse"; p.fuse_lora(lora_scale=1.0)
        stage = "unload"; p.unload_lora_weights()        # weights stay fused; removes the extra LoRA layers
        stage = "verify"
        after = (tr.blocks[0].attn1.to_q.weight.detach().float().sum().item(),
                 None if vace_w is None else tr.vace_blocks[0].proj_out.weight.detach().float().sum().item())
        if after[0] == before[0]:
            raise RuntimeError("the LoRA was loaded but the base weights did not change (nothing was fused)")
        if before[1] is not None and abs(after[1] - before[1]) > 1e-3 * max(abs(before[1]), 1.0):
            raise RuntimeError("the LoRA changed VACE's layout-control blocks (it must not)")
        stage = "scheduler"; p.scheduler = turbo_scheduler()
    except Exception as e:
        e.turbo_stage = stage
        raise
    log(f"turbo LoRA fused and verified in {time.time() - t0:.1f}s ({path.name}); scheduler {type(p.scheduler).__name__}")

def speed_up(p):
    """Apply VIDEO_SPEED. Returns the settings the clips are made with. Never stops the run: on any failure it
    logs the full error, SHOWS the reason, and falls back to the next slower mode (recorded in each clip's sidecar)."""
    mode = VIDEO_SPEED
    if mode == "turbo" and VIDEO_ENGINE != "wan21-1.3b":
        log("turbo LoRA only exists for Wan 2.1 1.3B; using fast mode")
        mode = "fast"
    turbo_error = None
    if mode == "turbo":
        try:
            fuse_turbo(p)
            print(f"Speed: turbo ({TURBO_STEPS} steps, Self-Forcing distilled model, verified)")
            return {"speed": "turbo", "steps": TURBO_STEPS, "guidance": 1.0}
        except Exception as e:
            log_exc("turbo LoRA failed, using fast mode")
            turbo_error = f"{getattr(e, 'turbo_stage', '?')}: {type(e).__name__}: {str(e)[:300]}"
            print(f"⚠️ Turbo mode FAILED at step {turbo_error}\n   → using fast mode instead (~6x slower per clip). "
                  "Full traceback in the log: search 'turbo LoRA failed'.")
            mode = "fast"
            try:                                           # a half-loaded adapter must not stay in the model
                p.unload_lora_weights()
            except Exception:
                pass
    if mode == "fast":
        try:
            from diffusers.hooks import FirstBlockCacheConfig
            p.transformer.enable_cache(FirstBlockCacheConfig(threshold=CACHE_THRESHOLD))
            print(f"Speed: fast ({CLIP_STEPS} steps + First Block Cache, threshold {CACHE_THRESHOLD})")
            return {"speed": "fast", "steps": CLIP_STEPS, "guidance": 5.0, "turbo_error": turbo_error}
        except Exception:
            log_exc("First Block Cache failed, using quality mode")
    print(f"Speed: quality ({CLIP_STEPS} full steps)")
    return {"speed": "quality", "steps": CLIP_STEPS, "guidance": 5.0, "turbo_error": turbo_error}

def weights_gb(path, kw, default):
    """Size of a model's weights at the dtype it is loaded in (what a CPU-side load holds in RAM at its peak)."""
    try:
        d = Path(path) / kw.get("subfolder", "")
        files = list(d.glob("*.safetensors")) if d.is_dir() else []
        if files:
            return sum(f.stat().st_size for f in files) / 2**30
    except Exception:
        pass
    return default

def load_to_gpu(cls, path, ram_gb=4.5, **kw):
    """Load a model straight onto the GPU (system RAM stays free). If that fails, the fallback goes through system
    RAM: only after checking there is room (ram_gb = its size there; VACE in half precision ≈ 3.6 GB), one weight at
    a time (low_cpu_mem_usage), then moved to the GPU at once. Never the unchecked full-RAM load that crashed Colab."""
    ram(f"(before loading {cls.__name__})")
    try:
        m = cls.from_pretrained(path, device_map={"": 0}, **kw)
        ram(f"({cls.__name__} on the GPU)")
        return m
    except Exception:
        log_exc(f"{cls.__name__}: direct GPU load failed, loading through system RAM")
    need_ram(weights_gb(path, kw, ram_gb), f"{cls.__name__} (loaded through system RAM)")
    m = cls.from_pretrained(path, low_cpu_mem_usage=True, **kw)
    if GPU["name"]:
        m = m.to("cuda")
    free_gpu()
    ram(f"({cls.__name__} moved to the GPU)")
    return m

def video_parts(mid, dt, tr_cls=None, name=None):
    """Transformer + VAE of a 1.3B model (Wan2.1 T2V, or VACE). From the Drive model cache when it is there (half
    precision, no download); otherwise from Hugging Face (full precision), then saved to the cache for next time."""
    import torch
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    tr_cls = tr_cls or WanTransformer3DModel
    cdir = MODEL_CACHE / f"{name or VIDEO_ENGINE}-{str(dt).split('.')[-1]}"
    if (cdir / "complete").exists():
        try:
            t0 = time.time()
            tr = load_to_gpu(tr_cls, cdir / "transformer", torch_dtype=dt)
            vae = load_to_gpu(AutoencoderKLWan, cdir / "vae", ram_gb=0.6, torch_dtype=torch.float32)
            print(f"Video model loaded from the model cache in {time.time() - t0:.0f}s (no download)")
            return tr, vae
        except Exception:
            log_exc("model cache unreadable, downloading again")
            shutil.rmtree(cdir, ignore_errors=True)
    need_disk(8, "the video model download (6.2 GB)", purge=[(TEXT_ENCODER_MIDS, ["text_encoder"])])
    tr = load_to_gpu(tr_cls, mid, subfolder="transformer", torch_dtype=dt)
    vae = load_to_gpu(AutoencoderKLWan, mid, ram_gb=0.6, subfolder="vae", torch_dtype=torch.float32)
    try:
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        drive_free = shutil.disk_usage(MODEL_CACHE).free / 2**30
        log(f"model cache: {drive_free:.1f} GB free where it is stored ({MODEL_CACHE})")
        if drive_free > 4:                              # the cache itself is ~3.3 GB
            t0 = time.time()
            tr.save_pretrained(cdir / "transformer", max_shard_size="1GB")   # 1 GB at a time: low RAM
            vae.save_pretrained(cdir / "vae")
            (cdir / "complete").write_text(mid)
            print(f"Saved the video model to the model cache in {time.time() - t0:.0f}s: next sessions skip the download")
            purge_hf([mid], ["transformer", "vae"])  # the model is in GPU memory and in the cache: free the disk
        else:
            print(f"⚠️ Only {drive_free:.1f} GB free on Google Drive: the video model cache (3.3 GB) was NOT saved, so "
                  "every new session downloads the model again (~4 min). Free up 4 GB on Drive to fix this.")
    except Exception:
        log_exc("could not save the model cache")
        shutil.rmtree(cdir, ignore_errors=True)
    return tr, vae

ZERO_LATENTS = MODEL_CACHE / "vae_zero_latents"   # ~17 MB per clip length; kept on Drive across rounds and sessions

def vae_guard(vae, ident="wan21-vae"):
    """Wrap a Wan VAE's encode/decode (VACE runs 2 encodes + 1 decode of the whole clip per generation, in fp32,
    OUTSIDE the step progress bar):
      - timing: synchronized seconds per call → vae.timing (shown per clip, written to perf.json);
      - VAE_TILING="auto": untiled first (tiles overlap, ~2.5x the pixels at 832x480); on out-of-memory the SAME call
        is retried with tiles and tiles stay on — the clip is never lost; True = always tiles (the old behaviour);
      - VAE_ZERO_CACHE: VACE's "inactive" video is all zeros (our mask covers the whole frame), so its latent is the
        same for every clip of that size: computed once, saved, then reused (bit-identical, one VAE pass saved per
        clip; the file survives the VACE reloads between rounds and the next sessions)."""
    if getattr(vae, "_guarded", False):
        return vae
    import torch
    enc, dec = vae.encode, vae.decode
    vae.timing = {"encode_s": 0.0, "decode_s": 0.0, "encode_calls": 0, "decode_calls": 0, "zero_reused": 0,
                  "tiles_after_oom": False, "last_encode_end": None}
    if VAE_TILING is True:
        vae.enable_tiling()

    def run(fn, kind, x, *a, **kw):
        t0 = cuda_now()
        try:
            out = fn(x, *a, **kw)
        except torch.cuda.OutOfMemoryError:
            if getattr(vae, "use_tiling", False):
                raise
            log(f"VAE {kind} without tiles: out of GPU memory ({gpu_mem()}) → tiles from now on", "WARN")
            free_gpu(); vae.enable_tiling(); vae.timing["tiles_after_oom"] = True
            out = fn(x, *a, **kw)
        t1 = cuda_now()
        vae.timing[f"{kind}_s"] += t1 - t0; vae.timing[f"{kind}_calls"] += 1
        if kind == "encode":
            vae.timing["last_encode_end"] = t1
        return out

    def encode(x, *a, **kw):
        if VAE_ZERO_CACHE and torch.is_tensor(x) and not a and kw.get("return_dict", True) and not bool(x.any()):
            tiles = bool(getattr(vae, "use_tiling", False))
            key = hashlib.sha1(f"{ident}|{tuple(x.shape)}|{x.dtype}|tiles={tiles}".encode()).hexdigest()[:16]
            f = ZERO_LATENTS / f"{key}.pt"
            if f.exists():
                try:
                    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
                    from diffusers.models.modeling_outputs import AutoencoderKLOutput
                    h = torch.load(f).to(x.device)
                    vae.timing["zero_reused"] += 1; vae.timing["last_encode_end"] = cuda_now()
                    return AutoencoderKLOutput(latent_dist=DiagonalGaussianDistribution(h))
                except Exception:
                    log_exc(f"cached zero latent {f.name} unreadable, encoding again")
            out = run(enc, "encode", x, *a, **kw)
            try:
                ZERO_LATENTS.mkdir(parents=True, exist_ok=True)
                torch.save(out.latent_dist.parameters.detach().cpu(), f)
            except Exception:
                log_exc("could not save the zero latent (computed again next time)")
            return out
        return run(enc, "encode", x, *a, **kw)

    vae.encode, vae.decode = encode, (lambda z, *a, **kw: run(dec, "decode", z, *a, **kw))
    vae._guarded = True
    return vae

def load_video_model(mode):
    """Video model WITHOUT the text encoder (prompts are pre-encoded)."""
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    mid = VIDEO_MID[VIDEO_ENGINE]
    dt = torch.bfloat16 if real_bf16() else torch.float16
    if mode == "vace":                   # layout-controlled generation for scene-state shots (any GPU)
        from diffusers import WanVACEPipeline, WanVACETransformer3DModel
        tr, vae = video_parts(VACE_MID, dt, WanVACETransformer3DModel, "wan21-vace-1.3b")
        p = WanVACEPipeline.from_pretrained(VACE_MID, transformer=tr, vae=vae, text_encoder=None, torch_dtype=dt)
        p.to("cuda")
        vae_guard(p.vae, f"{VACE_MID}|{p.vae.dtype}")   # timed; untiled first, tiles only after out-of-memory; zero-latent reuse
        p.speed = speed_up(p)            # turbo LoRA fuses into VACE's base blocks (verified in fuse_turbo)
        ram("(VACE layout model loaded)")
        return p
    if mode == "i2v":
        from diffusers import WanImageToVideoPipeline as cls
    else:
        cls = WanPipeline
    if VIDEO_ENGINE == "wan22-5b":
        vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)
        p = cls.from_pretrained(mid, vae=vae, text_encoder=None, torch_dtype=dt)
        p.enable_model_cpu_offload()     # 5B: L4/A100 sessions have 50–80 GB RAM, so offloading is safe
    else:                                # 1.3B (~3 GB) sits on the T4 GPU: keeps system RAM free
        tr, vae = video_parts(mid, dt)
        # transformer and VAE are passed in, so only the small config files are downloaded
        p = cls.from_pretrained(mid, transformer=tr, vae=vae, text_encoder=None, torch_dtype=dt)
        p.to("cuda")
    p.speed = speed_up(p)
    ram("(video model loaded)")
    return p

CLIP_WRITER = 2   # 1 = diffusers export_to_video (bug: it multiplies 8-bit frames by 255 -> photo-negative colours)

def write_clip(arr, fps, path):
    """8-bit RGB frames (frames, H, W, 3) -> near-lossless H.264 file. Our own writer: diffusers' export_to_video
    expects 0..1 floats and turned our 0..255 frames into a colour negative, and compressed them heavily."""
    assert arr.dtype == np.uint8 and arr.ndim == 4 and arr.shape[-1] == 3, (arr.dtype, arr.shape)
    n, h, w, _ = arr.shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "12",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    t0 = time.time()
    r = subprocess.run(cmd, input=np.ascontiguousarray(arr).tobytes(), capture_output=True)
    log(f"clip written ({time.time() - t0:.1f}s, exit {r.returncode}): {n} frames {w}x{h} -> {Path(path).name}", "FFMPG")
    if r.returncode:
        raise RuntimeError("writing the clip failed: " + r.stderr.decode(errors="replace")[-2000:])

def check_clip_writer():
    """Self-test: known colours in, same colours out. Stops the run instead of making wrong-coloured videos."""
    ref = np.zeros((8, 64, 64, 3), np.uint8)
    ref[:, :, :32] = (200, 80, 30)                   # fox orange
    ref[:, :, 32:] = (20, 30, 60)                    # night blue
    p = WORK / "writer_test.mp4"
    write_clip(ref, 16, p)
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    back = np.frombuffer(raw, np.uint8).reshape(-1, 64, 64, 3)
    err = np.abs(back[:, 8:24, 8:24].astype(int).mean((0, 1, 2)) - [200, 80, 30]).max()
    err = max(err, np.abs(back[:, 8:24, 40:56].astype(int).mean((0, 1, 2)) - [20, 30, 60]).max())
    log(f"clip writer self-test: max colour error {err:.1f}")
    assert err < 6, f"clip writer changes colours (error {err:.1f}); not making any clips"

# ---- continuity check: is every character who should be on screen actually in the clip? ----
# CLIP (openai/clip-vit-base-patch32, MIT, ~0.6 GB) compares 8 frames of the clip with "a scene with <character>"
# and with "the same scene with nobody in it". A character counts as seen in a frame unless the second clearly wins.
PRESENCE_MODEL, PRESENCE_MIN = "openai/clip-vit-base-patch32", 0.75   # seen in at least 6 of 8 frames
PRESENCE_FRAME_P = 0.2      # per frame: below this the character is clearly missing (tested: absent ≈ 0.01, present ≈ 0.5)
_presence = None

def presence_model():
    global _presence
    if _presence is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if GPU["name"] else "cpu"
        dt = torch.float16 if dev == "cuda" else torch.float32
        src = str(hf_files(PRESENCE_MODEL, "continuity checker",     # real files on Drive (no broken symlinks);
                           allow=["*.json", "*.txt", "pytorch_model.bin"]))  # PyTorch weights only (not TF/Flax copies)
        kw = dict(torch_dtype=dt)
        try:                             # straight onto the GPU: no copy in system RAM
            m = CLIPModel.from_pretrained(src, device_map={"": 0} if dev == "cuda" else None, **kw).eval()
        except Exception:
            log_exc("CLIP: direct GPU load failed, loading through system RAM")
            need_ram(1.0, "the CLIP continuity checker")
            m = CLIPModel.from_pretrained(src, low_cpu_mem_usage=True, **kw).to(dev).eval()
        _presence = (m, CLIPProcessor.from_pretrained(src), dev, dt)
        log(f"continuity checker loaded ({PRESENCE_MODEL} on {dev})")
    return _presence

def presence(arr, tags, scene):
    """{tag: {"seen": share of sampled frames where it is visible, "mean": average probability}}."""
    import torch
    m, proc, dev, dt = presence_model()
    idx = np.linspace(0, len(arr) - 1, 8).round().astype(int)
    # the two descriptions differ ONLY in whether the character is there (tag first: CLIP reads at most 77 tokens)
    texts = [f"{t}, in {scene}" for t in tags] + [f"{scene}, with nobody in it"]
    inp = proc(text=texts, images=[Image.fromarray(arr[i]) for i in idx], return_tensors="pt",
               padding=True, truncation=True)
    inp = {k: (v.to(dev, dt) if k == "pixel_values" else v.to(dev)) for k, v in inp.items()}
    with torch.no_grad():
        logits = m(**inp).logits_per_image.float().cpu()                     # (8 frames, tags + 1)
    out = {}
    for j, t in enumerate(tags):
        p = torch.softmax(logits[:, [j, -1]], dim=1)[:, 0].numpy()
        out[t] = {"seen": round(float((p > PRESENCE_FRAME_P).mean()), 3), "mean": round(float(p.mean()), 3)}
    return out

def continuity_score(pres):
    if not pres:
        return 1.0
    return min(v["seen"] for v in pres.values()) + 0.01 * min(v["mean"] for v in pres.values())

RELEVANCE_MIN = 0.5          # the clip must match its OWN scene better than any other place in ≥ half the frames

def relevance(arr, own, others):
    """Narration ↔ visual check at render time: share of sampled frames where CLIP matches the shot's own scene
    description better than every scene description from OTHER places in the video."""
    import torch
    if not others:
        return None
    m, proc, dev, dt = presence_model()
    idx = np.linspace(0, len(arr) - 1, 8).round().astype(int)
    inp = proc(text=[own] + others, images=[Image.fromarray(arr[i]) for i in idx], return_tensors="pt",
               padding=True, truncation=True)
    inp = {k: (v.to(dev, dt) if k == "pixel_values" else v.to(dev)) for k, v in inp.items()}
    with torch.no_grad():
        logits = m(**inp).logits_per_image.float().cpu()                     # (8 frames, 1 + others)
    win = (logits.argmax(dim=1) == 0).float()
    return {"seen": round(float(win.mean()), 3), "mean": round(float(torch.softmax(logits, 1)[:, 0].mean()), 3)}

def distractors(shot):
    """Scene descriptions of AI shots set somewhere else (same-place shots are allowed to look alike)."""
    here = shot.get("location", shot.get("scene"))
    out = []
    for s in SHOTS:
        if s.get("motion_prompt") and s.get("location", s.get("scene")) != here and s.get("scene") not in out:
            out.append(s["scene"])
    return out[:12]

def missing_from(pres):
    return [t for t, v in (pres or {}).items()
            if v["seen"] < (RELEVANCE_MIN if t.startswith("scene:") else PRESENCE_MIN)]

def frames_for(dur, fps, most, max_stretch=1.15, fewest=33):
    """Frames to generate for a shot of `dur` seconds: enough to cover it when slowed down at most 1.15x.
    Generating fewer frames is much faster (attention cost grows with the square of the frame count).
    Wan needs 4k+1 frames."""
    f = math.ceil(dur * fps / max_stretch) + 1
    f = 4 * math.ceil((f - 1) / 4) + 1
    return max(fewest, min(most, f))

def make_clip(pipe_v, mode, shot, image, part, emb, frames=None):
    import torch
    fps, nf, w, h = NATIVE[VIDEO_ENGINE]
    nf = frames or nf
    torch.manual_seed(shot["seed"] + part)            # turbo re-noises between steps with the global generator
    kw = dict(prompt_embeds=emb[clip_prompt(shot, mode)].to("cuda"), negative_prompt_embeds=emb[NEG_V].to("cuda"),
              height=h, width=w, num_frames=nf, num_inference_steps=pipe_v.speed["steps"],
              guidance_scale=pipe_v.speed["guidance"], generator=torch.Generator("cpu").manual_seed(shot["seed"] + part))
    log(f"{shot['shot_id']} part {part}: {pipe_v.speed} {w}x{h} {nf} frames")
    if mode == "i2v":
        kw["image"] = ImageOps.fit(image.convert("RGB"), (w, h))
    # frames stay on the GPU as a tensor and become 8-bit there: 4x less system RAM than a float32 array
    vid = pipe_v(**kw, output_type="pt").frames[0]                  # (frames, 3, H, W), values 0..1
    if not torch.isfinite(vid).all():
        raise RuntimeError("the model produced a broken clip (NaN)")
    arr = (vid.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    del vid
    if arr.mean() < 2.5:
        raise RuntimeError("the model produced a black clip")
    return arr, fps, nf

PERF = {}

def timing_line(tm):
    """'(VAE encode 1.2 + 25 steps x 30.1 s + VAE decode 2.3 min, GPU 97% busy)' for the screen."""
    if not tm.get("step_avg_s"):
        return ""
    parts = [f"VAE in {tm.get('vae_encode_s', 0) / 60:.1f}", f"{tm.get('steps')} steps x {tm['step_avg_s']:.0f} s",
             f"VAE out {tm.get('vae_decode_s', 0) / 60:.1f} min"]
    util = f", GPU {tm['gpu_util_avg']:.0f}% busy" if tm.get("gpu_util_avg") is not None else ""
    return "(" + " + ".join(parts) + util + ")"

def perf_report():
    """Where the scene-state time went: per clip and overall. Screen table + render/perf.json (for comparisons)."""
    if not PERF.get("clips"):
        return
    PERF["total_s"] = round(time.time() - PERF["t0"], 1)
    PERF["checker_loads"] = CHECKER_LOADS.get("n", 0)
    keys = ["layout_video_s", "vae_encode_s", "denoise_s", "vae_decode_s", "to_frames_s", "write_s", "check_s"]
    print("\nWhere the time went (seconds; GPU-synchronized):")
    print("  shot  rnd speed  steps " + " ".join(f"{k[:-2]:>12}" for k in keys) + "  s/step  GPU%  decision")
    for c in PERF["clips"]:
        print(f"  {c['shot']:5} {c['round']:>3} {str(c.get('speed')):6} {str(c.get('steps')):>5} "
              + " ".join(f"{c.get(k, 0) or 0:12.0f}" for k in keys)
              + f"  {c.get('step_avg_s') or 0:6.1f}  {c.get('gpu_util_avg') or 0:4.0f}  {c['decision']}")
    print(f"  model loads: VACE x{PERF['vace_loads']} ({PERF['vace_load_s']:.0f}s), checkers x{PERF['checker_loads']} "
          f"({CHECKER_LOADS.get('s', 0):.0f}s), releases {PERF['release_s']:.0f}s | rounds {PERF['rounds']} | "
          f"regenerations {PERF['regenerations']}/{PERF['generations']} | total {PERF['total_s'] / 60:.1f} min")
    try:
        (PROJ / "render").mkdir(parents=True, exist_ok=True)
        (PROJ / "render" / "perf.json").write_text(json.dumps(PERF, indent=2, default=str))
    except Exception:
        log_exc("could not write perf.json")
    log("perf: " + json.dumps(PERF, default=str)[:4000])

CLIPS_T0 = time.time()
def gpu_budget_spent():
    """MAX_GPU_MINUTES: once clip making has run this long, no new clip is started (the rest use their fallback)."""
    used = (time.time() - CLIPS_T0) / 60
    if used >= globals().get("MAX_GPU_MINUTES", 1e9):
        log(f"GPU budget used: {used:.0f} of {MAX_GPU_MINUTES} min; no new clips", "WARN")
        return True
    return False

video_shots = [s for s in SHOTS if s.get("motion_prompt")]
logic_all = [s for s in video_shots if logic_shot(s)]
def _logic_state(s):
    sc = read_sidecar(PROJ / "clips" / s["shot_id"]) or {}
    if sc.get("gen_key") != logic_gen_key(s) or not all((PROJ / "clips" / p).exists() for p in sc.get("parts", ["-"])):
        return "generate"
    return "ok" if sc.get("val_key") == logic_val_key(s) else "recheck"
logic_todo = [s for s in logic_all if _logic_state(s) == "generate"]
logic_recheck = [s for s in logic_all if _logic_state(s) == "recheck"]
if BENCHMARK:
    run_benchmark()          # times one shot under each setting, writes render/bench/benchmark.md, then stops
todo = [s for s in video_shots if not logic_shot(s) and clip_info(s) is None]
if logic_recheck:
    print(f"Re-checking {len(logic_recheck)} cached clips because the plan or the checks changed: "
          f"{[s['shot_id'] for s in logic_recheck]}")
if todo and VIDEO_ENGINE is None:
    print("No GPU for video: these shots will be camera moves over stills:", [s["shot_id"] for s in todo])
elif todo:
    check_clip_writer()
    bad = [s["shot_id"] for s in todo if (read_sidecar(PROJ / "clips" / s["shot_id"]) or {}).get("writer", 2) < CLIP_WRITER]
    if bad:
        print(f"Remaking {len(bad)} clips that were saved with inverted colours by the old writer: {bad}")
    for s in video_shots:                            # shots finished in earlier runs: edit them while the GPU works
        if s not in todo:
            edit_in_background(s)
    pipe_v, mode, emb = None, VIDEO_MODE, None
    wait_for_prefetch()
    try:
        emb = encode_prompts(sorted({clip_prompt(s, mode) for s in todo} | {NEG_V}))
        pipe_v = load_video_model(mode)
        print(f"Video model loaded ({VIDEO_ENGINE}, {mode}). {len(todo)} shots to make "
              f"(each shows a {pipe_v.speed['steps']}-step progress bar; finished shots are saved to Drive as they complete).")
    except Exception as e:
        log_exc("video model could not load")
        print("❌ Video model could not load:", repr(e)[:500])
    spent = []
    for i, shot in enumerate(todo, 1):
        if pipe_v is None:
            break
        if gpu_budget_spent():
            print(f"⏱️ MAX_GPU_MINUTES ({MAX_GPU_MINUTES}) reached: the remaining {len(todo) - i + 1} shots use stills/fallbacks")
            break
        sid, t0 = shot["shot_id"], time.time()
        fps, nf = NATIVE[VIDEO_ENGINE][:2]
        # A long line gets more clips (max 3) instead of a frozen last frame:
        # - image-to-video: each clip continues from the last frame of the one before (one continuous shot)
        # - text-to-video with characters: ONE clip only. A second clip would be an unrelated new take and the
        #   characters would jump, change or vanish mid-shot (story plans are checked to fit one clip)
        # - text-to-video, scenery only (space, maps...): extra takes are fine: a cut between two views of a nebula
        #   looks natural, and nothing can "vanish"
        parts_needed = max(1, min(3, math.ceil(TL[sid]["dur"] / (nf / fps * MAX_SLOWMO))))
        if mode != "i2v" and shot.get("must_show"):
            parts_needed = 1
        tags = list(shot.get("must_show") or []) if CONTINUITY_CHECK else []
        frames = frames_for(TL[sid]["dur"], fps, nf) if (FRAMES_ON_DEMAND and parts_needed == 1) else None
        img = Image.open(PROJ / "keyframes" / f"{sid}.png") if mode == "i2v" else None
        names = []
        try:
            checks = []
            for part in range(1, parts_needed + 1):
                best = None
                for attempt in range(1 + (CONTINUITY_RETRIES if (tags or CONTINUITY_CHECK) else 0)):
                    arr, fps, nf = make_clip(pipe_v, mode, shot, img, part + 1000 * attempt, emb, frames)
                    pres = None
                    if tags:
                        try:
                            pres = presence(arr, tags, shot.get("scene", shot["image_prompt"]))
                        except Exception:
                            log_exc("continuity check failed; it is switched off for this run")
                            tags = []
                    if CONTINUITY_CHECK and shot.get("scene"):
                        try:                                  # does the clip show THIS scene, not another one?
                            rel = relevance(arr, shot["scene"], distractors(shot))
                            if rel is not None:
                                pres = dict(pres or {}, **{"scene: " + shot["scene"][:40]: rel})
                        except Exception:
                            log_exc("scene relevance check failed")
                    log(f"{sid} part {part} attempt {attempt + 1}: presence {pres}")
                    if best is None or continuity_score(pres) > continuity_score(best[1]):
                        best = (arr, pres, attempt)
                    del arr
                    people = [m for m in missing_from(pres) if not m.startswith("scene:")]
                    if not people:                    # scene-only doubts: flag for review, don't spend GPU time
                        break
                    if attempt < CONTINUITY_RETRIES:
                        print(f"    {sid}: not on screen enough: {people} → remaking with a new seed")
                arr, pres, attempt = best; best = None
                checks.append({"part": part, "presence": pres, "attempts": attempt + 1, "missing": missing_from(pres)})
                if missing_from(pres):
                    print(f"    ⚠️ {sid}: the check isn't sure these are visible: {missing_from(pres)}"
                          + (" → its drawn fallback scene will be used" if shot.get("fallback") else " → kept the best take")
                          + f"; look at render/review/{sid}.jpg")
                name = f"{sid}_p{part}.mp4"
                write_clip(arr, fps, WORK / name)
                shutil.copy(WORK / name, PROJ / "clips" / name)
                names.append(name)
                img = Image.fromarray(arr[-1])
                del arr; free_gpu()
            sidecar(PROJ / "clips" / sid, **clip_key(shot, mode), parts=names, fps=fps, frames_per_part=nf,
                    writer=CLIP_WRITER, continuity=checks,
                    use_fallback=bool(shot.get("fallback")) and any(c["missing"] for c in checks), engine=VIDEO_ENGINE, mode=mode, steps=pipe_v.speed["steps"], speed=pipe_v.speed["speed"], gpu=GPU["name"],
                    seconds=round(time.time() - t0, 1))
            edit_in_background(shot)                  # cut + grade this shot on the CPU while the GPU goes on
            spent.append(time.time() - t0)
            left = sum(spent) / len(spent) * (len(todo) - i) / 60
            print(f"  [{i}/{len(todo)}] {sid}: {len(names)} clip(s) in {spent[-1] / 60:.1f} min"
                  f" | about {left:.0f} min left for the other shots")
            ram()
        except Exception as e:
            log_exc(f"{sid}: video failed")
            print(f"  [{i}/{len(todo)}] {sid}: video failed ({repr(e)[:300]}); this shot will use a camera move")
            free_gpu()
    release_model(pipe_v, "video model"); release_model((_presence or [None])[0], "CLIP")
    del pipe_v, emb; _presence = None; free_gpu()

# ---- scene-state shots: VACE layout control, every clip checked against the plan, targeted repairs ----
# Runs in ROUNDS so the 15 GB T4 only ever holds one model family: (A) VACE makes a candidate for every pending
# shot → VACE released; (B) the checkers (OWLv2 + Qwen3-VL) check every candidate → checkers released; shots that
# need a targeted repair go to the next round (at most REPAIR_BUDGET more rounds).
if (logic_todo or logic_recheck) and VIDEO_ENGINE is None:
    print("No GPU: scene-state shots can't be generated or checked now; they use their fallback until a GPU run:",
          [s["shot_id"] for s in logic_todo + logic_recheck])
elif logic_todo or logic_recheck:
    import torch
    TRACK = {s["shot_id"]: {"params": {"cs": 1.0, "skip": set(), "static": False, "seed_off": 0, "frames": logic_frames(s)},
                            "tried": [], "attempts": [], "best": None, "t0": time.time(), "gen_error": None}
             for s in logic_todo}
    pending, emb = list(logic_todo), None
    for _s in pending:
        for _w in precheck(_s):
            print(f"  ⚠️ plan check {_s['shot_id']}: {_w}"); log(f"plan check {_s['shot_id']}: {_w}", "WARN")
    PERF.update(t0=time.time(), vace_loads=0, vace_load_s=0.0, checker_load_s=0.0, release_s=0.0, rounds=0,
                generations=0, regenerations=0, clips=[])
    if pending:
        check_clip_writer()
        wait_for_prefetch()
        try:
            emb = encode_prompts(sorted({logic_prompt(s) for s in pending} | {logic_prompt(s, True) for s in pending}
                                        | {NEG_V}))
            log(f"GPU after the prompts: {gpu_mem()}")
        except Exception as e:
            log_exc("prompts could not be encoded"); print("❌ Prompts could not be encoded; scene-state shots use their fallback")
            pending = []

    def finish(shot):
        """Write the final clip + sidecar for a shot whose rounds are over."""
        sid, T = shot["shot_id"], TRACK[shot["shot_id"]]
        if T["best"] is None:
            print(f"  {sid}: no usable candidate ({T['gen_error'] or 'not generated'}); its fallback will be used")
            return
        path, fps, n, results, dec, _ = T["best"]
        final = dec if dec.status == "ACCEPT" else acceptance.decide(results, T["tried"], 0)
        name = f"{sid}_p1.mp4"
        shutil.copy(path, PROJ / "clips" / name)
        if final.status != "ACCEPT":
            review_assets(shot, decode_clip(path), results, final)
        sidecar(PROJ / "clips" / sid, gen_key=logic_gen_key(shot), val_key=logic_val_key(shot), decision=final.status,
                parts=[name], fps=fps, frames_per_part=n, writer=CLIP_WRITER, engine="wan21-vace-1.3b", mode="vace",
                gpu=GPU["name"], checks=[r.to_dict() for r in results], attempts=T["attempts"], repairs=T["tried"],
                failed=final.reasons, uncertain=final.unknown, seconds=round(time.time() - T["t0"], 1),
                motion_prompt=shot["motion_prompt"], image_prompt=shot["image_prompt"], seed=shot["seed"],
                speed=T.get("speed"))
        mark = "✅ accepted" if final.status == "ACCEPT" else f"⚠️ {final.status}: not used unless approved"
        print(f"  {sid}: {mark} after {len(T['attempts'])} attempt(s)"
              + ("" if final.status == "ACCEPT" else f" | see render/review/{sid}.*"))
        if final.status == "ACCEPT":
            edit_in_background(shot)

    rnd = 0
    while pending:
        rnd += 1
        # (A) generation: only VACE on the GPU
        try:
            tl = time.time()
            pipe_v = load_video_model("vace")
            PERF["vace_loads"] += 1; PERF["vace_load_s"] += time.time() - tl; PERF["rounds"] = rnd
            log(f"round {rnd}: VACE loaded in {time.time() - tl:.0f}s, GPU {gpu_mem()}")
            if rnd == 1:
                print(f"VACE layout model loaded. {len(pending)} scene-state shots; each is checked against the plan "
                      f"(positions, direction, stops, hand-offs, order) and repaired up to {REPAIR_BUDGET}x.")
        except Exception as e:
            log_exc("VACE model could not load")
            print("❌ VACE model could not load (details in the log); these shots use their fallback:", repr(e)[:300])
            break
        cands = {}
        for shot in pending:
            sid, T = shot["shot_id"], TRACK[shot["shot_id"]]
            if gpu_budget_spent():                        # no clip → finish() gives the shot its drawn fallback
                REVIEW[sid] = f"not made: MAX_GPU_MINUTES ({MAX_GPU_MINUTES}) reached"
                continue
            t0 = time.time()
            try:
                arr, fps = make_vace_clip_safe(pipe_v, shot, emb, T["params"])
                path = WORK / f"cand_{sid}_{len(T['attempts']) + 1}.mp4"
                tw = time.time(); write_clip(arr, fps, path)
                tm = dict(CLIP_TIMING.get(sid, {}), write_s=round(time.time() - tw, 1), clip_total_s=round(time.time() - t0, 1))
                CLIP_TIMING[sid] = tm; T["speed"] = pipe_v.speed.get("speed")
                PERF["generations"] += 1; PERF["regenerations"] += rnd > 1
                cands[sid] = (path, fps, len(arr))
                del arr
                print(f"  round {rnd} · {sid}: clip made in {(time.time() - t0) / 60:.1f} min " + timing_line(tm))
            except Exception as e:
                log_exc(f"{sid}: video failed"); T["gen_error"] = repr(e)[:200]
                print(f"  round {rnd} · {sid}: video failed ({repr(e)[:200]})")
            free_gpu()
        tr0 = time.time(); release_model(pipe_v, "VACE"); del pipe_v; PERF["release_s"] += time.time() - tr0
        # (B) checking: only the checkers on the GPU
        next_pending = []
        for shot in pending:
            sid, T = shot["shot_id"], TRACK[shot["shot_id"]]
            if sid not in cands:
                finish(shot)
                continue
            path, fps, n = cands[sid]
            tc = time.time(); results = logic_checks(shot, decode_clip(path), fps); check_s = round(time.time() - tc, 1)
            dec = acceptance.decide(results, T["tried"], REPAIR_BUDGET)
            n_fail = sum(r.status == "FAIL" and r.critical for r in results)
            tm = dict(CLIP_TIMING.get(sid, {}), check_s=check_s)
            PERF["clips"].append(dict(tm, shot=sid, round=rnd, decision=dec.status, repair=dec.repair))
            T["attempts"].append({"params": {k: (sorted(v) if isinstance(v, set) else v) for k, v in T["params"].items()},
                                  "decision": dec.status, "failed": dec.reasons, "uncertain": dec.unknown, "timing": tm})
            log(f"{sid} attempt {len(T['attempts'])}: {dec.status} {dec.repair or ''} | failed {dec.reasons} | "
                f"uncertain {dec.unknown}")
            best_ok = T["best"] is not None and T["best"][4].status == "ACCEPT"
            if not best_ok and (T["best"] is None or dec.status == "ACCEPT" or n_fail < T["best"][5]):
                T["best"] = (path, fps, n, results, dec, n_fail)    # fewest critical failures (for review only)
            if dec.status == "REPAIR":
                print(f"    {sid}: {dec.reasons[0] if dec.reasons else 'check failed'} → repair: {dec.repair}")
                T["params"] = apply_repair(dec.repair, T["params"], STATES[sid]); T["tried"].append(dec.repair)
                next_pending.append(shot)
            else:
                finish(shot)
        tr0 = time.time(); release_checkers(); PERF["release_s"] += time.time() - tr0
        pending = next_pending
        ram()
    for shot in pending:                              # rounds stopped early (model failed to load)
        finish(shot)
    for shot in logic_recheck:                        # cached clip, changed plan/checks: check again, no regeneration
        sid = shot["shot_id"]
        try:
            sc = read_sidecar(PROJ / "clips" / sid)
            arr = decode_clip(PROJ / "clips" / sc["parts"][0])
            results = logic_checks(shot, arr, sc.get("fps", 16))
            final = acceptance.decide(results, [], 0)
            if final.status != "ACCEPT":
                review_assets(shot, arr, results, final)
            sc.update(val_key=logic_val_key(shot), decision=final.status, checks=[r.to_dict() for r in results],
                      failed=final.reasons, uncertain=final.unknown)
            sidecar(PROJ / "clips" / sid, **sc)
            print(f"  {sid}: re-checked → {final.status}")
        except Exception:
            log_exc(f"{sid}: re-check failed")
            print(f"  {sid}: re-check failed (details in the log); its fallback is used")
    release_checkers()
    del emb; free_gpu()
    log(f"scene-state shots done, GPU {gpu_mem()}")
    perf_report()
ready = sum(clip_info(s) is not None for s in video_shots)
make_keyframes([s for s in SHOTS if not (s.get("motion_prompt") and clip_info(s))      # fallback pictures, if any
                and not (logic_shot(s) and s.get("fallback"))])                     # (drawn fallback needs none)
print(f"{ready}/{len(video_shots)} shots have real AI video."
      + ("" if ready == len(video_shots) else "  Re-run this notebook later to make the missing ones."))
