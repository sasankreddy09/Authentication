# Run the Telugu 3D video in Google Colab

Use [this video notebook](https://colab.research.google.com/github/sasankreddy09/Authentication/blob/feat/cinematic-storybook-characters-v2/colab/notebooks/colab_three_d_friends_te_v3_storybook.ipynb). It is self-contained. You do not need to upload the repository, website, Python source tree or model weights to Drive.

## What you add to Drive

```text
MyDrive/ai-video/
  references/                       # needed once for Character Studio
    arjun.png                       # supplied boy reference, actual image file
    meera.png                       # supplied girl reference, actual image file
  _assets/
    characters/
      arjun/cinematic-v2/            # COMPLETE approved atlas version
        manifest.json
        ...all referenced images and previews...
      meera/cinematic-v2/            # COMPLETE approved atlas version
        manifest.json
        ...all referenced images and previews...
    environments/
      cinema/v1/                    # copy this whole folder from colab/assets/environments/
      lobby/v1/                     # copy this whole folder from colab/assets/environments/
    studio/reference-v2-candidate-01/ # created by Studio; retain reviewed inputs and approval records
```

The character folders must come from the **approved cinematic Studio output**, not the repository's old `v1` procedural examples. The reference images alone are insufficient: animation also needs poses, blink/mouth/expression patches, manifest mappings and hashes. Copy entire version folders without renaming internal files. If you choose other version names, set `CHARACTER_VERSIONS` in Settings to match. Do not edit an approved version in place; create a new version and project.

**Already have approved packs?** Add the two character folders and the two environment folders. You do not need to regenerate characters. **Only have the pictures?** Run [Character Studio](../cinematic/README.md) first, review the generated design and actual animation, correct cutouts/placement as needed, and approve the full packs. The repository does not currently supply finished approved cinematic characters. Model-based artwork generation and its Colab memory requirements remain to be measured on your allocated runtime.

If your originals are JPEGs, either convert them to PNG or set the actual `.jpg` paths in the Studio notebook. Renaming a file is not image conversion. The pictures attached in ChatGPT are not automatically copied into Google Drive.

## One-time account setup

1. Sign into Hugging Face and open [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts). Accept its access terms and wait for access if required.
2. Create a token with read access. In Colab's key icon (**Secrets**), add it as `HF_TOKEN` and enable notebook access. Keep the token out of Drive files, notebook source and GitHub.
3. For the separate image-generation workflow, follow any model-access steps reported by Character Studio. No Gemini, OpenAI or paid API key is required by this Telugu render target.

## Run and resume

Select **Runtime → Change runtime type → T4 GPU**, if Colab offers one. Open Settings, check character versions and `PROJECT`, then choose **Run all** and authorize the Drive mount. Preflight stops with the exact missing folder/resource before speech generation.

The default is 720p/24 fps, a 120-second measured-video limit, up to two ordinary speech attempts, and a 60-minute local session budget. GPU handles new speech and encoding when NVENC is available; CPU handles animation, text and the small multilingual speech diagnostic. Large image/video generation is excluded from this notebook. Hardware and free quota are not guaranteed. Do not leave other models loaded in the same runtime.

The notebook creates these folders; **you do not upload them for a first run**:

```text
MyDrive/ai-video/three_d_friends_te_v3_cinematic/
  plan.json           # generated, bound to the actual artwork hashes
  narration/          # WAV files and JSON cache/word-timing sidecars
  diagrams/           # cached animation clips
  segments/           # cached edited clips
  render/             # final.mp4, subtitles, metadata and QC reports
  logs/               # run log and speech-worker diagnostics
```

Keep the complete project folder to resume. After a disconnect, reconnect and Run all. On a CPU runtime, set `RENDER_CACHED_AUDIO_ONLY=True` only if that project already contains all matching WAVs **and their `.wav.json` sidecars**; stale/missing speech stops instead of synthesizing on CPU. It can still download the small speech-check model if required.

Large voice/model caches use **Colab local scratch by default**, avoiding several GB of free Drive consumption. A fresh runtime may download them again. The optional `MODEL_CACHE_ON_DRIVE` setting applies to the shared model/font cache; the isolated Parler worker's Hugging Face cache remains local. Leave Drive room for the complete artwork and rendered outputs. Actual usage depends on atlas resolution and retained previews; the notebook does not know your account's remaining storage quota reliably.

Listen to the Telugu narration and inspect facial patches, gestures, scene continuity and subtitles in `final.mp4`. Small Whisper is a diagnostic and is less reliable on Telugu/code mixing; flagged or unavailable checks require review. Neither generating the file nor saving the notebook uploads anything to YouTube.

Speaker-identity model checks (ECAPA) and their extra retakes are omitted from the free-tier target to avoid loading another model alongside Parler. The QC report records that check as UNKNOWN; compare both voices by listening.
