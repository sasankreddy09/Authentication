"""Expressive English narration: Qwen3-TTS CustomVoice, run in its OWN virtual environment.

qwen-tts pins transformers==4.57.3; the notebook needs 5.x. So, like parler_worker.py, this file is started once by
the narration cell with the TTS environment's python, keeps the model loaded, and answers one JSON request per line:

  request:  {"text": "...", "speaker": "Ryan", "language": "English", "instruct": "Speak with quiet curiosity...",
             "seed": 123, "out": "/path/x.wav", "gen": {"temperature": 0.9}}
  response: {"ok": true, "out": "...", "seconds": 3.2, "gen_s": 4.1}  |  {"ok": false, "error": "..."}

The first line is {"ready": true, "sr": ..., "device": ..., "dtype": ..., "speakers": [...], "instruct": bool}.
Precision is a parameter: a T4 has no native bf16 and Qwen3-TTS has reported fp16 instability on older GPUs, so the
notebook benchmarks fp32/fp16 instead of assuming. Attention: PyTorch SDPA (FlashAttention-2 is never required).

Run: python qwen_tts_worker.py Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice cuda float32
"""
import json, sys, time

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"   # Apache-2.0; instruction control (the 0.6B has none)
INSTRUCT_MODELS = {"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
# Everything qwen-tts 0.1.1 needs on top of Colab, resolved for Python 3.13 / x86_64 Linux
# (uv pip compile --python-version 3.13 --python-platform x86_64-manylinux_2_28), then torch, torchaudio, triton,
# numpy, nvidia-* and cuda-* REMOVED (Colab's matching copies are used). Installed with --no-deps.
REQUIREMENTS = """
accelerate==1.12.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
audioop-lts==0.2.2
brotli==1.2.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
decorator==5.3.1
einops==0.8.2
fastapi==0.143.3
filelock==4.2.1
flatbuffers==25.12.19
fsspec==2026.9.0
gradio==6.17.3
gradio-client==2.5.0
groovy==0.1.2
h11==0.16.0
hf-gradio==0.4.1
hf-xet==1.7.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.36.2
idna==3.20
jinja2==3.1.6
joblib==1.6.0
lazy-loader==0.6
librosa==1.0.0
llvmlite==0.50.0
markdown-it-py==4.2.0
markupsafe==3.0.4
mdurl==0.1.2
mpmath==1.3.0
msgpack==1.2.3
narwhals==2.27.1
networkx==3.7
numba==0.68.0
onnxruntime==1.31.0
opentelemetry-api==1.45.1
orjson==3.13.0
packaging==26.3
pandas==3.0.6
pillow==12.3.0
platformdirs==4.13.4
pooch==1.9.0
protobuf==7.36.2
psutil==7.2.2
pycparser==3.11
pydantic==2.14.1
pydantic-core==2.50.1
pydub==0.25.1
pygments==2.21.0
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytz==2026.5
pyyaml==6.0.3
qwen-tts==0.1.1
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safehttpx==0.1.7
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.18.1
semantic-version==2.10.0
setuptools==84.0.0
shellingham==1.5.4
six==1.17.0
soundfile==0.14.0
sox==1.5.0
soxr==1.1.0
starlette==1.8.0
sympy==1.14.0
threadpoolctl==3.7.0
tokenizers==0.22.2
tomlkit==0.14.0
tqdm==4.70.1
transformers==4.57.3
typer==0.27.3
typing-extensions==4.16.0
typing-inspection==0.4.4
urllib3==2.8.0
uvicorn==0.54.0
"""


def main():
    mid, device, dtype = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    proto, sys.stdout = sys.stdout, sys.stderr        # library prints must never mix into the JSON channel
    def say(obj):
        proto.write(json.dumps(obj) + "\n"); proto.flush()
    import numpy as np, soundfile as sf, torch
    from qwen_tts import Qwen3TTSModel
    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = Qwen3TTSModel.from_pretrained(mid, device_map=("cuda:0" if device == "cuda" else "cpu"), dtype=dt,
                                          attn_implementation="sdpa")
    try:
        speakers = list(model.get_supported_speakers())
    except Exception:
        speakers = []
    say({"ready": True, "device": device, "dtype": dtype, "load_s": round(time.time() - t0, 1), "speakers": speakers,
         "instruct": mid in INSTRUCT_MODELS, "transformers": __import__("transformers").__version__,
         "vram_gb": round(torch.cuda.memory_allocated() / 2**30, 2) if device == "cuda" else None})
    for raw in sys.stdin:
        if not raw.strip():
            continue
        try:
            req = json.loads(raw)
            t1 = time.time()
            torch.manual_seed(int(req.get("seed", 0)))
            if device == "cuda":
                torch.cuda.manual_seed_all(int(req.get("seed", 0))); torch.cuda.reset_peak_memory_stats()
            kw = dict(text=req["text"], language=req.get("language", "English"), speaker=req["speaker"])
            if req.get("instruct"):
                kw["instruct"] = req["instruct"]
            with torch.inference_mode():
                wavs, sr = model.generate_custom_voice(**kw, **(req.get("gen") or {}))
            audio = np.asarray(wavs[0], dtype=np.float32).squeeze()
            if audio.size == 0 or not np.isfinite(audio).all():
                raise RuntimeError("the voice model produced empty or broken audio (try float32 on this GPU)")
            sf.write(req["out"], audio, sr)
            say({"ok": True, "out": req["out"], "seconds": round(audio.size / sr, 2), "sr": sr,
                 "gen_s": round(time.time() - t1, 1),
                 "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else None})
        except Exception as e:
            import traceback
            traceback.print_exc(file=sys.stderr)
            say({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})


if __name__ == "__main__":
    main()
