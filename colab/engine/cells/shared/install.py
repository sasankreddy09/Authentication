# ==== 2. Install packages (1–2 min the first time; skipped when this Colab machine already has them) ====
import sys, os, importlib.util; print("Python", sys.version.split()[0])
_MARK = "/content/.ai_video_packages_v5"        # v5: + librosa/speechbrain 1.1.2 (voice QC, narrator identity)
if os.path.exists(_MARK):
    print("Packages already installed on this machine: skipped.")
else:
    # Colab's preinstalled torchao is older than new diffusers expects ("cannot import name 'FqnToConfig'").
    # This notebook doesn't use torchao, so remove it.
    !pip -q uninstall -y torchao
    # kokoro-onnx = the Kokoro voice model without PyTorch; works on Colab's newer Python
    # exact versions this notebook was tested with (all have Python 3.13 wheels); newer releases can change APIs
    !pip -q install kokoro-onnx==0.6.1 soundfile==0.14.0 faster-whisper==1.2.1 ctranslate2==4.8.3 ftfy imageio==2.38.1 imageio-ffmpeg==0.6.0 || pip -q install --ignore-requires-python kokoro-onnx==0.6.1 soundfile==0.14.0 faster-whisper==1.2.1 ctranslate2==4.8.3 ftfy imageio==2.38.1 imageio-ffmpeg==0.6.0
    !pip -q install diffusers==0.41.0 transformers==5.19.0 accelerate==1.15.0 peft==0.21.2 huggingface_hub==1.33.0 safetensors==0.8.0
    !pip -q install "pydantic>=2.7" scipy librosa speechbrain==1.1.2
    # GPU build of onnxruntime so the Kokoro voice can run on the GPU (both builds share one module, so swap them)
    !nvidia-smi -L > /dev/null 2>&1 && pip -q uninstall -y onnxruntime && pip -q install onnxruntime-gpu && echo "onnxruntime-gpu installed"
    if all(importlib.util.find_spec(m) for m in ["kokoro_onnx", "faster_whisper", "diffusers", "peft", "imageio_ffmpeg",
                                                 "pydantic", "scipy", "librosa", "speechbrain"]):
        open(_MARK, "w").write("ok")
    else:
        print("⚠️  Some packages did not install; this cell will try again next time.")
if "diffusers" in sys.modules or "torchao" in sys.modules:
    print("⚠️  Old libraries are still loaded in memory. Click Runtime → Restart session, then Runtime → Run all.")
