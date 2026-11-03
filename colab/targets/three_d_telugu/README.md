# How 3D movies work — Telugu, cinematic character edition

This is the **only video notebook upgraded to the new characters**. It uses your reviewed Arjun and Meera artwork, the existing Telugu dialogue and voices, cinema/lobby scenes, and measured speech timing.

Before Run all:
1. Prepare and approve both character atlases with the separate Character Studio workflow. Upload the complete `characters/arjun/cinematic-v2/` and `characters/meera/cinematic-v2/` folders to `MyDrive/ai-video/_assets/`. Reference pictures alone cannot animate the video.
2. Copy `colab/assets/environments/cinema/v1/` and `colab/assets/environments/lobby/v1/` into the same `_assets/environments/` hierarchy.
3. Accept the Indic Parler model terms at https://huggingface.co/ai4bharat/indic-parler-tts and add a read token named `HF_TOKEN` in **Colab Secrets**, with notebook access enabled.
4. Select an allocated **T4 GPU** in Runtime → Change runtime type. Check Settings, then Run all.

Output: `MyDrive/ai-video/three_d_friends_te_v3_cinematic/render/final.mp4`, subtitles, metadata and QC reports. The notebook saves a plan with the exact approved artwork hashes. If you replace artwork or change the story, choose a new `PROJECT` folder; an existing project cannot silently switch assets.

**Free-tier preset:** 1280×720 at 24 fps, one drawing worker, up to two speech attempts per line, a 120-second finished-video limit and a 60-minute local session budget. Parler uses GPU; small multilingual Whisper uses CPU int8; Pillow animation/text use CPU; FFmpeg tries NVENC and falls back to CPU. No image or video diffusion models run here. Colab quota, hardware availability and session duration vary; these limits do not guarantee completion in one session. Large model weights use local scratch by default.

If disconnected, reconnect and Run all; completed matching narration and render segments are reused. CPU-only rendering is available with `RENDER_CACHED_AUDIO_ONLY=True` once this project has all matching narration WAVs **and their JSON sidecars**. A missing/stale line stops instead of starting a large CPU speech model.

Read `docs/GOOGLE_COLAB.md` in the repository for the Drive folder tree and first-time artwork workflow. Inspect the rendered characters and listen to the Telugu audio before sharing. CPU integration tests do not certify reference fidelity or Colab CUDA compatibility.

### Current script revision: v4

The notebook now embeds `colab/stories/three_d_friends_te_v4/story.json` via its updated plan. Default output is `MyDrive/ai-video/three_d_friends_te_v4_cinematic/`. Only cinema/v1 is required; the earlier lobby instruction applies to v3. The new script stays in the cinema, uses original Telugu humor, and selects CPU-generated curiosity music. Review the new dialogue/audio before sharing; previous script approvals do not apply.

### Current script revision: v5 (retention edit)

The notebook now uses the v5 story and writes to `MyDrive/ai-video/three_d_friends_te_v5_cinematic/`. The main question opens the video; the eye and glasses diagrams remain visible through their explanations; the one-eye qualification follows the mechanism; a two-line joke closes the video. This supersedes the v4 output-path note above. Character packs, cinema background and local Parler requirements remain the same. Actual speech duration, naturalness and audience retention still need verification.


Speech-check interpretation: Telugu scores are normalized character/code-mix
recognizer disagreement, not measured percentages of wrongly pronounced words.
The recognizer model now follows ASR_MODEL / ASR_MODEL_MULTI. `small` remains the
low-memory default; `large-v3-turbo` is available for a stronger diagnostic but
requires more GPU memory and is not guaranteed accurate on a particular clip.
Whisper documents language-dependent performance:
https://github.com/openai/whisper#available-models-and-languages

ASR_RETRY_TELUGU defaults to False: a high ASR mismatch alone flags a line instead
of generating successive takes to optimize against potentially incorrect transcripts.
This does not waive QC or change thresholds. Signal-based retakes remain separate.
New takes save expected text, TTS input, heard transcript, score and device under
narration/speech_checks. Existing cached audio/scores are retained; their presence
is not evidence that a new model rechecked them. Listen to a flagged WAV and compare
the transcript before attributing disagreement to either the TTS or recognizer.
