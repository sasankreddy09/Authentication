# ==== 8d. Benchmark (only when BENCHMARK = True in the settings) ====
# Times ONE scene-state shot (BENCH_SHOT) under each configuration in BENCH_CONFIGS: same shot, seed, frame count and
# prompts, so the numbers are comparable. Every result is checked by the same validators (QC decision = quality
# proxy) and saved for a visual comparison. Writes render/bench/benchmark.md + benchmark.json, then stops the
# notebook (no video is made). Set BENCHMARK = False afterwards for normal runs.
BENCH = {
    "baseline":    {"speed": "fast", "vae_tiling": True, "zero_cache": False,
                    "about": "the old setting: 25 steps x 2 passes (guidance) + First Block Cache, VAE in tiles"},
    "turbo_tiled": {"speed": "turbo", "vae_tiling": True, "zero_cache": False,
                    "about": "turbo LoRA (4 steps, 1 pass) only; VAE still in tiles"},
    "optimized":   {"speed": "turbo", "vae_tiling": "auto", "zero_cache": True,
                    "about": "turbo + untiled VAE (tiles only after out-of-memory) + zero-latent reuse"},
}


class BenchmarkDone(Exception):
    """Raised on purpose after a benchmark so 'Run all' stops before making the video."""


def run_benchmark():
    global VIDEO_SPEED, VAE_TILING, VAE_ZERO_CACHE
    import torch
    shot = next((s for s in SHOTS if s["shot_id"] == BENCH_SHOT), None)
    if shot is None or not logic_shot(shot):
        print(f"BENCHMARK: {BENCH_SHOT} is not a scene-state shot of this plan; choose one of {sorted(STATES)}")
        return
    if VIDEO_ENGINE is None:
        print("BENCHMARK needs a GPU runtime"); return
    out = PROJ / "render" / "bench"; out.mkdir(parents=True, exist_ok=True)
    keep = (VIDEO_SPEED, VAE_TILING, VAE_ZERO_CACHE)
    print(f"Benchmark of {BENCH_SHOT} ({logic_frames(shot)} frames, {VACE_NATIVE[2]}x{VACE_NATIVE[3]}) on {GPU['name']}: "
          f"{', '.join(BENCH_CONFIGS)}")
    emb = encode_prompts(sorted({logic_prompt(shot), NEG_V}))
    rows = []
    for name in BENCH_CONFIGS:
        cfg = BENCH[name]
        VIDEO_SPEED, VAE_TILING, VAE_ZERO_CACHE = cfg["speed"], cfg["vae_tiling"], cfg["zero_cache"]
        row = {"config": name, "about": cfg["about"]}
        pipe = None
        try:
            free_gpu(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            pipe = load_video_model("vace")
            row["load_s"] = round(time.time() - t0, 1)
            row["speed_used"], row["turbo_error"] = pipe.speed["speed"], pipe.speed.get("turbo_error")
            if cfg["speed"] == "turbo" and pipe.speed["speed"] != "turbo":
                row["note"] = "turbo failed to load: not measured (see turbo_error)"
                print(f"  {name}: turbo failed ({row['turbo_error']}); skipped")
                continue
            torch.cuda.reset_peak_memory_stats()
            params = {"cs": 1.0, "skip": set(), "static": False, "seed_off": 0, "frames": logic_frames(shot)}
            t1 = time.time()
            arr, fps = make_vace_clip(pipe, shot, emb, params)
            path = out / f"{BENCH_SHOT}_{name}.mp4"
            write_clip(arr, fps, path)
            row["clip_total_s"] = round(time.time() - t1, 1)
            row["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            row.update(CLIP_TIMING.get(BENCH_SHOT, {}))
            row["path"] = str(path)
            Image.fromarray(np.concatenate([arr[i] for i in np.linspace(0, len(arr) - 1, 4).round().astype(int)],
                                           axis=1)).resize((1664, 240)).save(out / f"{BENCH_SHOT}_{name}.jpg", quality=85)
            print(f"  {name}: {row['clip_total_s'] / 60:.1f} min for the clip {timing_line(row)}, "
                  f"peak GPU memory {row['peak_vram_gb']} GB")
        except Exception as e:
            log_exc(f"benchmark {name} failed"); row["error"] = repr(e)[:300]
            print(f"  {name}: FAILED {repr(e)[:200]}")
        finally:
            rows.append(row)
            if pipe is not None:
                release_model(pipe, "VACE (benchmark)"); del pipe
            free_gpu()
    VIDEO_SPEED, VAE_TILING, VAE_ZERO_CACHE = keep
    for row in rows:                                   # same checks for every result: quality is compared, not assumed
        if row.get("path"):
            try:
                res = logic_checks(shot, decode_clip(Path(row["path"])), VACE_NATIVE[0])
                d = acceptance.decide(res, [], 0)
                row.update(qc=d.status, qc_failed=d.reasons, qc_uncertain=d.unknown)
            except Exception as e:
                log_exc("benchmark check failed"); row["qc"] = f"check failed: {repr(e)[:100]}"
    release_checkers()
    base = next((r for r in rows if r["config"] == "baseline" and r.get("clip_total_s")), None)
    cols = [("Model initialisation (s)", "load_s"), ("Diffusion (s)", "denoise_s"), ("Seconds per step", "step_avg_s"),
            ("VAE encode (s)", "vae_encode_s"), ("VAE decode (s)", "vae_decode_s"), ("Total clip (s)", "clip_total_s"),
            ("Peak GPU memory (GB)", "peak_vram_gb"), ("Average GPU utilisation (%)", "gpu_util_avg"),
            ("QC decision", "qc")]
    md = [f"# Benchmark: {BENCH_SHOT} on {GPU['name']} ({time.strftime('%Y-%m-%d %H:%M')})", "",
          "| Measurement | " + " | ".join(r["config"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for label, k in cols:
        md.append(f"| {label} | " + " | ".join(str(r.get(k, "—")) for r in rows) + " |")
    if base:
        md.append("| Speed-up vs baseline (total clip, measured) | " + " | ".join(
            f"{base['clip_total_s'] / r['clip_total_s']:.1f}x" if r.get("clip_total_s") else "—" for r in rows) + " |")
    md += ["", *[f"- **{r['config']}**: {r['about']}" + (f" — {r.get('note') or r.get('error')}" if r.get('note') or r.get('error') else "")
                 for r in rows], "", "Clips and 4-frame strips for a visual comparison are in this folder."]
    (out / "benchmark.md").write_text("\n".join(md))
    (out / "benchmark.json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n".join(md))
    raise BenchmarkDone(f"Benchmark finished: {out / 'benchmark.md'}. Set BENCHMARK = False to make the video.")
