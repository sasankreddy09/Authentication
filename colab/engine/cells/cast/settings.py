# ==== 1. Settings (you can edit these) ====
NOTEBOOK_VERSION = "story v5 (faster: frames on demand + background downloads; continuity: one continuous clip per shot, plan check + on-screen character check; disk/RAM guard; colour fix; model + edit caches; turbo AI video; full run log)"
print("Notebook version:", NOTEBOOK_VERSION)
PROJECT     = "__PROJECT__"                  # folder name on Google Drive
DRIVE_ROOT  = "/content/drive/MyDrive/ai-video"
VOICE       = "__VOICE__"                    # only used to test the voice model; each character's voice is in the plan
SPEED_SCALE = 1.0                            # multiplies every line's speed (0.95 = a bit slower overall)
WER_RETRY   = 0.15                           # re-make a line if the speech check finds >15% wrong words
ASR_MODEL   = "small.en"                     # speech-check / subtitle-timing model
EXPRESSIVE  = True                           # scene-aware delivery: each line gets a direction from the story
VOICE_PROFILE = "science_explainer"          # science_explainer | beginner_education | cinematic_documentary |
                                             # mystery_documentary | historical_narration | inspirational_story
VOICE_DIRECTOR = "rules"                     # "rules": plan tones + story roles (no model) | "llm": a local model
DIRECTOR_MODEL = "Qwen/Qwen3-4B-Instruct-2507"   # used only when VOICE_DIRECTOR = "llm"
VOICE_BACKEND = "kokoro"                     # "kokoro" (fast; follows pace + pauses only) | "qwen3-tts" (follows the
                                             # whole direction as an instruction; 1.7B, ~3 min setup; measure first
                                             # with VOICE_BENCH)
NARRATOR_QWEN_SPEAKER = "aiden"              # Qwen3-TTS narrator identity: "aiden" or "ryan" (native English)
EXPRESSIVE_MAX_INTENSITY = 0.5              # cap for instruction-following voices (measured: strong styles shifted
                                             # the narrator's timbre by ~5 semitones and changed a word)
QWEN_TTS_DTYPE = "float32"                   # T4 has no fast bf16; fp16 can be unstable on older GPUs (benchmark it)
VOICE_BENCH = False                          # True: neutral vs expressive narration A/B for each backend, then stop
MAX_VOICE_TAKES_NORMAL = 1                   # takes per ordinary line (a retry still happens on a technical failure)
MAX_VOICE_TAKES_HERO = 2                     # opening / central question / reveal / ending: 2 takes on Qwen3-TTS
                                             # or Parler (Kokoro is deterministic: one take), both kept for A/B
ASR_MODEL_MULTI = "large-v3-turbo"           # speech check for non-English videos (Telugu): multilingual Whisper
SPEECH_RETRY_OTHER = 0.35                    # non-English: re-make a line if > 35 % of its LETTERS are heard wrong
                                             # (unverified starting point: Whisper's Telugu is weak; see the log values)
PARLER_SEC_PER_CHAR = (0.025, 0.25)          # Telugu voice: a take outside this length per letter is cut off/rambling
VIDEO_SPEED = "turbo"                        # "turbo" (T4: ~3–5 min per shot), "fast" (~15 min), "quality" (~28 min)
CLIP_STEPS  = 25                             # steps per clip in "fast"/"quality"
CACHE_THRESHOLD = 0.08                       # "fast" mode only
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
BURN_SUBS   = True                           # burn coloured subtitles into the video (an .srt is saved either way)
SUB_STYLE   = "karaoke"                      # "karaoke": each word lights up as it is spoken (ASS); "plain": classic
MUSIC_GAIN  = 0.30                           # music level before ducking under the voices
DUCK = {"threshold": 0.02, "ratio": 8, "attack": 20, "release": 500}   # music ducking under the voice (sidechaincompress)
RUN_QC_STRICT = True                         # final report: approved only if EVERY mandatory check is PASS
SFX_GAIN_DB = 0.0                            # raise/lower all sound effects together
# ---- two-friend storytelling, licences and budgets (TARGET_DURATION = TARGET_SECONDS in the topic notebook;
#      MAX_REPAIR_ATTEMPTS = REPAIR_BUDGET above; RUN_MODE below) ----
STORY_MODE  = "__STORY_MODE__"               # "two_friends_cinematic": two original 2D friends talk it through |
                                             # "narrator": the classic narrated explainer (set by the plan)
VISUAL_STYLE = "stylized_2d"                 # the friends are drawn by code in a flat stylized 2D look (only style)
LICENSE_STRICT = __LICENSE_STRICT__          # True: refuse every model whose licence is marked "review" in the
                                             # registry (turbo LoRA → fast mode, SDXL → drawn stills, OpenAI CLIP check off)
LIP_SYNC    = "energy"                       # "energy": the mouth opens with the speaker's own loudness | "none": closed
MAX_AI_VIDEO_SHOTS = 2                       # story mode: AI video only where nothing cheaper is faithful
MAX_GPU_MINUTES = 90                         # no new AI clip is started after this many minutes of clip making
# ---- Telugu voices via Google Gemini FREE TIER (optional; the only hosted exception, for speech only) ----
STRICT_OPEN_SOURCE = __STRICT_OPEN_SOURCE__  # True: local open-source voices only (Indic Parler-TTS for Telugu);
                                             # Gemini (proprietary, even on the free tier) runs ONLY if you also set
                                             # TELUGU_TTS_BACKEND = "gemini_free" yourself — never via "auto"
TELUGU_TTS_BACKEND = "auto"                  # "auto": what the plan asks for | "gemini_free" | "indic_parler" (local)
GEMINI_TRANSCRIPT_FORM = "display"           # what Gemini reads: "display" (English words in English letters, as in the
                                             # subtitles) | "telugu_script" (respelled) — the listening benchmark decides
TELUGU_PRON_BENCH = False                    # True: one short exchange in every pronunciation form (Parler: respelled vs
                                             # Latin; Gemini, if enabled: display vs respelled) → blind listening sheet
GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"    # or "gemini-3.8-flash-lite-tts" (only if listening says it is as good)
GEMINI_FREE_TIER_CONFIRMED = False           # True ONLY after AI Studio → Projects shows this key's project with
                                             # Billing Tier "Free" (no billing account). A key alone proves nothing.
GEMINI_FREE_TIER_ONLY = True                 # never a paid route (no Batch/Flex/Priority, no billed key)
GEMINI_MODE = "scene"                        # "scene": two-speaker conversation chunks | "turn": one request per line
MAX_VIDEOS_PER_DAY = 1                       # local safety budget (NOT Google's quota, which AI Studio shows)
MAX_VIDEO_DURATION_SECONDS = 120
MAX_GEMINI_TTS_CALLS_PER_VIDEO = 6
MAX_GEMINI_RETRIES_PER_SEGMENT = 1
TELUGU_VOICE_BENCH = False                   # True: Parler vs Gemini (scene / per-turn) on a short dialogue, blind A/B
TELUGU_VOICE_AUDITION = False                # True: Gemini voice candidates for Arjun and Meera on one Telugu exchange
TELUGU_BENCH_LITE = False                    # also try gemini-3.8-flash-lite-tts in the benchmark (1 more call)
MAX_GEMINI_BENCH_CALLS = 14                  # separate local budget for the benchmark / audition (one-off)
FPS, W, H   = 24, 1920, 1080
RUN_MODE    = "standard"                     # "standard": 1920x1080 | "draft": 1280x720 quick preview |
                                             # "cpu_fallback": 1280x720, no AI video (drawn / licensed images only)
if RUN_MODE in ("draft", "cpu_fallback"):
    W, H = 1280, 720
X264_PRESET = "veryfast"                     # final encode on CPU (if no GPU encoder)
MODEL_CACHE_ON_DRIVE = True                  # keep models on Drive (~4 GB) so new sessions skip ~6 GB of downloads
