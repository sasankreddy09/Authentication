"""Render character review movies through the existing drawkit and FFmpeg, using existing speech WAVs."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import numpy as np
from bootstrap import ROOT, materialize


def import_prototype(candidate, assets, version):
    import storybook as SB, storybook_atlas as SA
    for cid in ("arjun", "meera"):
        src = Path(candidate) / cid / "atlas_src"
        spec = json.loads((src / "spec.json").read_text())
        # These substitutions are only for a provisional two-pose motion experiment.
        spec["pose_map"] = {p: "point_front" if p == "point" else "idle_front" for p in SB.POSES}
        SA.import_atlas(src, assets, cid, version, spec,
            {"model": "black-forest-labs/FLUX.2-klein-4B", "prototype_only": True})


def render_review(assets, version, audio, text, output, approved=False, fps=24, size=(640,360), closeup=False):
    import storybook as SB, drawkit, cinematic_workflow as CW
    from PIL import Image
    assets, output = Path(assets), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["STORYBOOK_ASSETS"] = str(assets)
    plan = {"visual_style_version": "storybook_v1", "environments": {"cinema": "v1"},
            "character_assets": {}}
    for cid in ("arjun", "meera"):
        ch = SB.load(cid, version, assets)
        if ch.man.get("asset_type") != "atlas_v1": raise ValueError("Preview needs an actual atlas")
        plan["character_assets"][cid] = {"version": version, "status": ch.man["status"],
            "manifest_sha256": SB.manifest_sha(ch.dir / "manifest.json"),
            "content_sha256": SB.content_fingerprint(ch.man)}
    if approved:
        plan["cinematic_characters"] = True
        CW.validate_plan(plan, assets)
    if not (assets / 'environments/cinema/v1/manifest.json').is_file():
        # Retain the existing environment artwork, not a new generated background.
        SB.install_pack((ROOT / 'files/storybook_assets_b9347befa279.tar.gz').read_bytes(), assets)
    wavs=[]; durations=[]; samples=[]; sr=None
    for p in audio:
        with wave.open(str(p)) as f:
            if f.getsampwidth()!=2 or f.getnchannels()!=1: raise ValueError('Use mono PCM16 speech WAV files')
            if sr is not None and f.getframerate()!=sr: raise ValueError('WAV sample rates must match')
            sr=f.getframerate(); a=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2')
        samples.append(a); durations.append(len(a)/sr)
    joined=np.concatenate([samples[0], np.zeros(int(sr*.25),dtype=np.int16), samples[1]])
    with wave.open(str(output/'speech.wav'),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes(joined.tobytes())
    offsets=[0,durations[0]+.25]; total=len(joined)/sr; n=math.ceil(total*fps)
    speech={};visemes={}
    for i,cid in enumerate(('arjun','meera')):
        full=np.zeros(len(joined),dtype=np.float32);start=round(offsets[i]*sr)
        full[start:start+len(samples[i])]=samples[i]/32768
        # Existing spectral mouth function and per-frame loudness; only the active speaker moves.
        visemes[cid], energy = SB.visemes_from_audio(full,sr,fps,n_frames=n)
        speech[cid] = energy.tolist()
    spec={'world':'cinema_seats','shot':'closeup' if closeup else 'two_shot','subject':'arjun',
          'chars':{'arjun':{'gesture':'idle','expr':'neutral'},'meera':{'gesture':'idle','expr':'neutral'}},
          'changes':[{'t':.3,'chars':{'arjun':{'gesture':'point'}}},
                     {'t':.65,'chars':{'meera':{'expr':'surprised'}}}],
          '_speech':speech,'_visemes':visemes,'_dur':total,'_n':n,
          'camera':{'movement':'slow_push_in'}}
    w,h=size
    raw=output/'frames.mp4'
    proc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-pix_fmt','yuv420p',str(raw)],stdin=subprocess.PIPE)
    try:
        for frame in drawkit.render('character',spec,plan,n,fps,w,h): proc.stdin.write(frame.tobytes())
    finally:
        proc.stdin.close()
    if proc.wait()!=0: raise RuntimeError('FFmpeg frame encoding failed')
    def stamp(t):
        ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'
    subs=output/'subtitles.srt'
    subs.write_text('\n\n'.join(f'{i+1}\n{stamp(offsets[i])} --> {stamp(offsets[i]+durations[i])}\n{text[i]}' for i in range(2)))
    # Run in the output directory so the subtitle filter never interpolates an arbitrary path.
    subprocess.run(['ffmpeg','-v','error','-y','-i','frames.mp4','-i','speech.wav','-vf',
        'subtitles=subtitles.srt','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','animation.mp4'],
        cwd=output,check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(output/'animation.mp4'),'-vf',
        'fps=1','-frames:v','8',str(output/'frame-%02d.png')],check=True)
    evidence={'approved_artwork':approved,'plan':plan,'audio_sha256':[CW.sha(p) for p in audio],
              'movie_sha256':CW.sha(output/'animation.mp4'),'renderer':'existing drawkit -> charanim dispatch -> storybook_scene -> FFmpeg',
              'note':'Human visual inspection required; no image-generation module used.'}
    (output/'render-evidence.json').write_text(json.dumps(evidence,indent=2))
    return output/'animation.mp4'


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--assets',required=True);ap.add_argument('--version',required=True)
    ap.add_argument('--audio',nargs=2,required=True);ap.add_argument('--text',nargs=2,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--candidate')
    ap.add_argument('--approved',action='store_true');ap.add_argument('--closeup',action='store_true')
    a=ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        sys.path.insert(0,str(materialize(tmp)))
        if a.candidate: import_prototype(a.candidate,a.assets,a.version)
        print(render_review(a.assets,a.version,a.audio,a.text,a.output,a.approved,closeup=a.closeup))
