# ==== 3b. Engine library: validation, drawn scenes, music library ====
# These are ordinary Python modules (colab/engine/lib/ in the project), copied into the notebook so it stays a
# single file to upload. They are unit-tested outside Colab (colab/tests/).
import sys, importlib
LIB_DIR = WORK.parent / "_engine_lib"
LIB_DIR.mkdir(parents=True, exist_ok=True)
for _name, _src in json.loads(r"""__LIB_FILES__""").items():
    (LIB_DIR / _name).write_text(_src)
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))
import validate, drawkit, musiclib, scene_state, layout_control, motion_check, trackers, vlm_check, acceptance, qclib, subtitles, stages, voice_direction, expressive_tts, voice_qc, registry, characters, charanim, demos, story, storybook_art, storybook, envfactory, storybook_scene
for _m in (validate, drawkit, musiclib, scene_state, layout_control, motion_check, trackers, vlm_check, acceptance, qclib, subtitles, stages, voice_direction, expressive_tts, voice_qc, registry, characters, charanim, demos, story, storybook_art, storybook, envfactory, storybook_scene):
    importlib.reload(_m)
log(f"engine library loaded from {LIB_DIR}: validate, drawkit, musiclib")
