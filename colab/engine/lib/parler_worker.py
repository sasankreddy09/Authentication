"""Telugu (and other Indic) voices: Indic Parler-TTS, run in its OWN virtual environment.

parler-tts pins transformers==4.46.1; the notebook needs transformers 5.x (Qwen3-VL). So this file is NOT imported by
the notebook: the narration cell starts it once with the TTS environment's python, keeps it running, and sends one
JSON request per line on stdin; the model loads once.

  request:  {"text": "...", "description": "Lalitha speaks ...", "seed": 123, "out": "/path/x.wav"}
  response: {"ok": true, "out": "...", "seconds": 3.2, "gen_s": 4.1}  |  {"ok": false, "error": "..."}

The first line printed is {"ready": true, "sr": ..., "device": ..., "dtype": ..., "load_s": ...}. Everything else
(warnings, progress) goes to stderr, which the notebook writes to its log file.

Run: python parler_worker.py ai4bharat/indic-parler-tts cuda float32
"""
import json, sys, time

MODEL = "ai4bharat/indic-parler-tts"            # Apache-2.0; gated (free, auto-approved): needs a Hugging Face token
PARLER_GIT = "git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68"
# Telugu speakers the model was trained on (model card: "Prakash, Lalitha, Kiran"; recommended: Prakash, Lalitha)
SPEAKERS = {"te": ["Prakash", "Lalitha", "Kiran"]}
# Everything parler-tts needs on top of Colab, resolved for Python 3.13 / x86_64 Linux with
#   uv pip compile --python-version 3.13 --python-platform x86_64-manylinux_2_28  (parler-tts @ PARLER_GIT)
# then torch, torchaudio, triton, numpy, nvidia-* and cuda-* REMOVED: the environment sees Colab's own (matching) copies.
# Installed with --no-deps, so nothing else is downloaded or upgraded.
REQUIREMENTS = """
absl-py==2.5.1
argbind==0.3.9
asttokens==3.0.2
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
decorator==5.3.1
descript-audio-codec==1.0.0
descript-audiotools @ git+https://github.com/descriptinc/audiotools@348ebf2034ce24e2a91a553e3171cb00c0c71678
docstring-parser==0.18.0
einops==0.8.2
executing==2.3.0
ffmpy==1.0.0
filelock==4.2.0
fire==0.7.1
flatten-dict==0.5.0
fonttools==4.67.0
fsspec==2026.9.0
grpcio==1.84.0
hf-xet==1.7.0
huggingface-hub==0.36.2
idna==3.20
importlib-resources==7.1.0
ipython==9.17.1
ipython-pygments-lexers==1.1.1
jedi==0.20.1
jinja2==3.1.6
joblib==1.6.0
julius==0.2.8
kiwisolver==1.5.1
lazy-loader==0.6
librosa==1.0.0
llvmlite==0.50.0
markdown==3.11.1
markdown-it-py==4.2.0
markdown2==2.5.5
markupsafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.3
narwhals==2.27.1
networkx==3.7
numba==0.68.0
packaging==26.3
parler-tts @ git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68
parso==0.8.7
pexpect==4.9.0
pillow==13.0.0
platformdirs==4.13.3
pooch==1.9.0
prompt-toolkit==3.0.53
protobuf==4.25.9
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pycparser==3.11
pygments==2.21.0
pyloudnorm==0.2.0
pyparsing==3.3.3
pystoi==0.4.1
python-dateutil==2.9.0.post0
pyyaml==6.0.3
randomname==0.2.1
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
sentencepiece==0.2.2
setuptools==84.0.0
six==1.17.0
soundfile==0.14.0
soxr==1.1.0
stack-data==0.6.3
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
termcolor==3.3.0
threadpoolctl==3.7.0
tokenizers==0.20.3
torch-stoi==0.2.3
tqdm==4.70.1
traitlets==5.16.1
transformers==4.46.1
typing-extensions==4.16.0
urllib3==2.8.0
wcwidth==0.9.2
werkzeug==3.1.9
"""


def main():
    mid, device, dtype = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    proto, sys.stdout = sys.stdout, sys.stderr        # library prints must never mix into the JSON channel
    def say(obj):
        proto.write(json.dumps(obj) + "\n"); proto.flush()
    import numpy as np, soundfile as sf, torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer, set_seed
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = ParlerTTSForConditionalGeneration.from_pretrained(mid, torch_dtype=dt).to(device).eval()
    tok = AutoTokenizer.from_pretrained(mid)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = model.config.sampling_rate
    say({"ready": True, "sr": sr, "device": device, "dtype": dtype, "load_s": round(time.time() - t0, 1),
         "transformers": __import__("transformers").__version__})
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
            t1 = time.time()
            set_seed(int(req.get("seed", 0)))
            d = desc_tok(req["description"], return_tensors="pt").to(device)
            p = tok(req["text"], return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                     prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            audio = gen.float().cpu().numpy().squeeze()
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("the voice model produced empty or broken audio")
            sf.write(req["out"], audio.astype(np.float32), sr)
            say({"ok": True, "out": req["out"], "seconds": round(audio.size / sr, 2), "gen_s": round(time.time() - t1, 1)})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            say({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})


if __name__ == "__main__":
    main()
