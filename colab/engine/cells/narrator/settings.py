# ==== 1. Settings (you can edit these) ====
NOTEBOOK_VERSION = "v15 (diagram shots drawn without AI; faster: frames on demand + background downloads; continuity: one continuous clip per shot + character check; disk/RAM guard; FIX: inverted/negative colours in AI clips; sharper, less warped motion; model cache, lazy keyframes, background editing + edit cache; turbo video: 4-step distilled Wan on T4; voice GPU fix; full run log on Drive; half-size text encoder, prompt cache, time-left estimate)"
print("Notebook version:", NOTEBOOK_VERSION)
PROJECT     = "__PROJECT__"                  # folder name on Google Drive (from the plan's "project" / "project_id")
DRIVE_ROOT  = "/content/drive/MyDrive/ai-video"
VOICE       = "am_michael"                   # Kokoro stock voice (e.g. am_michael, am_adam, bf_emma, af_heart)
SPEED_SCALE = 1.0                            # multiplies every mood's voice speed (0.95 = a bit slower overall)
WER_RETRY   = 0.15                           # re-make a sentence if the speech check finds >15% wrong words
ASR_MODEL   = "small.en"                     # speech-check / subtitle-timing model (runs on CPU)
ASR_MODEL_MULTI = "large-v3-turbo"           # speech check for non-English videos (Telugu): multilingual Whisper
SPEECH_RETRY_OTHER = 0.35                    # non-English: re-make a line if > 35 % of its LETTERS are heard wrong
                                             # (unverified starting point: Whisper's Telugu is weak; see the log values)
PARLER_SEC_PER_CHAR = (0.025, 0.25)          # Telugu voice: a take outside this length per letter is cut off/rambling
VIDEO_SPEED = "turbo"                        # "turbo": 4-step distilled model (T4: ~3–4 min per shot)
                                             # "fast":  25 steps + First Block Cache (~1.5–2x faster than quality)
                                             # "quality": 25 full steps (T4: ~28 min per shot)
CLIP_STEPS  = 25                             # video model steps per clip in "fast"/"quality" (more = slower)
CACHE_THRESHOLD = 0.08                       # "fast" mode: higher = faster, but more risk of smeary motion
MAX_SLOWMO  = 1.4                            # a clip may be slowed down this much to fit a longer line; longer
                                             # lines get extra clips instead (heavy slow motion looks smeary)
SMOOTH_MOTION = "blend"                      # frame-rate change / slow motion: "blend" (soft, never warps),
                                             # "mci" (motion-interpolated: smoother but can warp shapes), "none"
SHARPEN     = 0.5                            # light sharpening after upscaling the small AI clips (0 = off)
FRAMES_ON_DEMAND = True                      # make only as many video frames as each shot needs (~1.6x faster)
CONTINUITY_CHECK = True                      # look at every clip: is each character who should be on screen there?
CONTINUITY_RETRIES = 1                       # if not, remake the clip with a new seed this many times (keeps the best)
# ---- logical consistency (shots with a scene-state "state" block in the plan) ----
VIDEO_RECIPE = "auto"                        # "auto": shots with a scene-state are generated with VACE layout control
                                             # (Wan2.1-VACE-1.3B: boxes from the plan steer positions and direction);
                                             # "t2v": plain text-to-video everywhere (no spatial control)
LOGIC_CHECK = True                           # after each clip: detect people/objects (OWLv2), measure camera motion,
                                             # check positions, direction, stops, hand-offs, order against the plan
VLM_CHECK = True                             # also ask a video-language model (Qwen3-VL-2B) if each action happened
REPAIR_BUDGET = 2                            # targeted repairs per shot before it goes to review
VAE_TILING = "auto"                          # VACE's VAE: "auto" = untiled (faster), tiles only after out-of-memory;
                                             # True = always tiles (old behaviour, slowest, lowest memory)
VAE_ZERO_CACHE = True                        # reuse VACE's all-zero "inactive" latent (bit-identical; one VAE pass less)
BENCHMARK = False                            # True: time ONE scene-state shot under each BENCH_CONFIGS setting, write
                                             # render/benchmark.md, then stop (no video is made). Takes ~15–40 min.
BENCH_SHOT = "sh11"                          # the shot to benchmark (a VACE walking scene is representative)
BENCH_CONFIGS = ["optimized", "turbo_tiled", "baseline"]   # baseline = the old fast mode, 25 steps (~25 min alone)
MUSIC_STYLE = "library"                      # "library": licensed tracks per chapter (credited in the description);
                                             # "curious" / "ambient": music generated in the notebook (offline fallback)
MUSIC_UNDER_VOICE_DB = 10                    # music bed this many dB under the voice (ducking adds ~6–10 dB while someone speaks)
BURN_SUBS   = True                           # burn subtitles into the video (an .srt is saved either way)
SUB_STYLE   = "karaoke"                      # "karaoke": each word lights up as it is spoken (ASS); "plain": classic
MUSIC_GAIN  = 0.30                           # music level before ducking under the voice
DUCK = {"threshold": 0.02, "ratio": 8, "attack": 20, "release": 500}   # music ducking under the voice (sidechaincompress)
RUN_QC_STRICT = True                         # final report: approved only if EVERY mandatory check is PASS
SFX_GAIN_DB = 0.0                            # raise/lower all sound effects together
LICENSE_STRICT = __LICENSE_STRICT__          # True: refuse models marked "review" in the licence registry
MAX_GPU_MINUTES = 90                         # no new AI clip is started after this many minutes of clip making
FPS, W, H   = 24, 1920, 1080
RUN_MODE    = "standard"                     # "standard": 1920x1080 | "draft": 1280x720 quick preview |
                                             # "cpu_fallback": 1280x720, no AI video (drawn / licensed images only)
if RUN_MODE in ("draft", "cpu_fallback"):
    W, H = 1280, 720
X264_PRESET = "veryfast"                     # final encode on CPU (if no GPU encoder): veryfast ≈ 3x faster than medium
MODEL_CACHE_ON_DRIVE = True                  # keep models on Drive (~4 GB) so new sessions skip ~6 GB of downloads
