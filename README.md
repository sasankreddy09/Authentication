# Telugu video automation

A Colab video pipeline with reusable Arjun/Meera character artwork, narration-driven animation, Telugu subtitles, music and FFmpeg rendering. The former Next.js/Supabase website has been removed from this branch; its history remains in Git.

**Start here:** [Google Colab and Drive setup](docs/GOOGLE_COLAB.md) · [Current architecture](docs/02_ARCHITECTURE.md) · [Evolution](docs/history/EVOLUTION.md)

The cinematic upgrade applies only to **How 3D movies work in Telugu**:

[Open the updated video notebook in Colab](https://colab.research.google.com/github/sasankreddy09/Authentication/blob/feat/cinematic-storybook-characters-v2/colab/notebooks/colab_three_d_friends_te_v3_storybook.ipynb)

Prepare the approved character atlases first using the [Character Studio workflow](cinematic/README.md). The supplied pictures are design references; the repository does not yet contain approved cinematic artwork. The old procedural packs remain available for historical videos. Other notebooks were relocated without changing their contents.

```text
colab/
  build_telugu.py          # builds only the upgraded Telugu video
  build.py                 # general builder for legacy plans
  engine/lib/              # editable Python engine, recovered from notebooks
  engine/cells/            # existing general builder templates
  targets/three_d_telugu/  # current target's editable cell sources
  notebooks/              # self-contained Colab notebooks
  plans/                   # per-video JSON inputs
  stories/                 # stories and grounding records
  assets/                  # characters, environments, historical packs
  reports/                 # historical previews and QC evidence
cinematic/                 # character design/import/review tooling and CPU tests
docs/                      # current guides, architecture, historical research and prompts
.github/workflows/         # CPU validation; no upload or deployment automation
```

Generate the notebook after editing its source, then run the checks:

```bash
python -m pip install -r requirements-dev.txt
python colab/build_telugu.py
python colab/build_telugu.py --check
python -m unittest discover -s cinematic/tests -v
python -m unittest discover -s colab/tests -v
```

FFmpeg and ffprobe must be on PATH for render tests. Tests use synthetic textures and cached synthetic audio, not approved character artwork. A real T4 run, visual comparison to the supplied references and Telugu listening review are still needed. Generation does not automatically publish to YouTube or publish a ChatGPT environment.
