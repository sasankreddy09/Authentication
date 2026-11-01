# ==== 1b. Run log ====
# Everything this notebook prints, every ffmpeg command, every error (with the full traceback), how long each
# cell took, and RAM/GPU use every 30 s go to a log file: MyDrive/ai-video/<project>/logs/run_<date>.log
# If something goes wrong, send that file to Claude.
import sys, os, time, threading, traceback, subprocess
from datetime import datetime

class RunLog:
    def __init__(self, path):
        self.lock = threading.Lock()
        self._open(path)
    def _open(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a", encoding="utf-8", buffering=1)
    def write(self, text, level="INFO"):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            for line in str(text).splitlines() or [""]:
                self.f.write(f"{stamp} {level:<5} {line}\n")
            self.f.flush()
    def move_to(self, path):
        """Continue the log in a new place (local disk → Google Drive once it is mounted)."""
        with self.lock:
            self.f.close()
            old = open(self.path, encoding="utf-8").read()
            os.remove(self.path)
            self._open(path)
            self.f.write(old); self.f.flush()

class Tee:
    """Copies everything written to the screen into the log (progress bars: only their final state)."""
    _is_tee = True
    def __init__(self, stream, level):
        self.stream, self.level, self.buf = stream, level, ""
    def write(self, s):
        self.stream.write(s)
        self.buf += s
        if "\n" in self.buf:
            *lines, self.buf = self.buf.split("\n")
            for ln in lines:
                ln = ln.split("\r")[-1].rstrip()
                if ln:
                    LOG.write(ln, self.level)
        self.buf = self.buf[self.buf.rfind("\r") + 1:]
        return len(s)
    def flush(self):
        self.stream.flush()
    def __getattr__(self, k):
        return getattr(self.stream, k)

def log(msg, level="INFO"):
    """Write to the log file only (not the screen)."""
    LOG.write(msg, level)

def log_exc(context):
    """Full traceback of the exception being handled → log file; the screen only shows a short message."""
    LOG.write(f"{context}\n{traceback.format_exc()}", "ERROR")

def resources():
    msg = ""
    try:
        import psutil
        vm = psutil.virtual_memory()
        msg += f"RAM {(vm.total - vm.available) / 2**30:.1f}/{vm.total / 2**30:.1f} GB"
    except ImportError:
        pass
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        used, tot, util = [float(x) for x in r.stdout.strip().splitlines()[0].split(",")]
        msg += f" | GPU {used / 1024:.1f}/{tot / 1024:.1f} GB, {util:.0f}% busy"
    except Exception:
        pass
    try:
        du = shutil.disk_usage("/content" if os.path.exists("/content") else "/")
        msg += f" | disk free {du.free / 2**30:.0f} GB"
    except Exception:
        pass
    return msg

import shutil
for name in ("stdout", "stderr"):                     # re-running this cell: unwrap the old copy first
    while getattr(getattr(sys, name), "_is_tee", False):
        setattr(sys, name, getattr(sys, name).stream)
if "LOG" not in globals():
    LOG = RunLog(f"/content/run_logs/run_{datetime.now():%Y%m%d_%H%M%S}.log" if os.path.exists("/content")
                 else os.path.join(os.environ.get("WORK_DIR", "."), "run_logs", f"run_{datetime.now():%Y%m%d_%H%M%S}.log"))
sys.stdout, sys.stderr = Tee(sys.stdout, "INFO"), Tee(sys.stderr, "WARN")

# RAM / GPU / disk every 30 s, to the log only (shows what was happening right before a crash)
if "_MONITOR_STOP" in globals():
    _MONITOR_STOP.set()
_MONITOR_STOP = threading.Event()
def _monitor(stop):
    """Every 30 s: resources + the highest system RAM use seen in between (sampled every 2 s, so short peaks that a
    30-second snapshot would miss are still logged)."""
    try:
        import psutil
    except ImportError:
        psutil = None
    peak, n = 0.0, 0
    while not stop.wait(2):
        if psutil is not None:
            vm = psutil.virtual_memory()
            peak = max(peak, (vm.total - vm.available) / 2**30)
            RUN_PEAKS["ram_gb"] = max(RUN_PEAKS["ram_gb"], round(peak, 2))
        n += 1
        if n % 5 == 0:                                   # every 10 s: GPU memory of ALL processes (voice workers too)
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                RUN_PEAKS["gpu_gb"] = max(RUN_PEAKS["gpu_gb"], round(float(r.stdout.split()[0]) / 1024, 2))
            except Exception:
                pass
        if n % 15 == 0:
            log(resources() + (f" | RAM peak {peak:.1f} GB in the last 30 s" if psutil is not None else ""), "RES")
            peak = 0.0
RUN_PEAKS = globals().get("RUN_PEAKS") or {"ram_gb": 0.0, "gpu_gb": 0.0}   # whole-run peaks, for the final report
DEVICE_REPORT = globals().get("DEVICE_REPORT") or {}                      # step → where it actually ran
threading.Thread(target=_monitor, args=(_MONITOR_STOP,), daemon=True).start()

# Each cell: start, end, how long it took, and the full error if it failed
try:
    _ip = get_ipython()
    for ev, fn in globals().get("_LOG_HOOKS", []):
        try:
            _ip.events.unregister(ev, fn)
        except ValueError:
            pass
    _CELL = {}
    def _pre(info):
        if not getattr(sys.stdout, "_is_tee", False):   # something replaced the screen output: re-attach
            sys.stdout = Tee(sys.stdout, "INFO")
        if not getattr(sys.stderr, "_is_tee", False):
            sys.stderr = Tee(sys.stderr, "WARN")
        title = next((l.strip("# =") for l in (info.raw_cell or "").splitlines() if l.strip()), "")[:80]
        _CELL.update(title=title, t0=time.time())
        log(f"──── CELL START: {title} | {resources()}", "CELL")
    def _post(result):
        dt = time.time() - _CELL.get("t0", time.time())
        err = result.error_in_exec or result.error_before_exec
        if err is not None:
            LOG.write(f"CELL FAILED after {dt:.1f}s: {_CELL.get('title')}\n"
                      + "".join(traceback.format_exception(type(err), err, err.__traceback__)), "ERROR")
        log(f"──── CELL END ({'error' if err else 'ok'}, {dt:.1f}s): {_CELL.get('title')} | {resources()}", "CELL")
    _LOG_HOOKS = [("pre_run_cell", _pre), ("post_run_cell", _post)]
    for ev, fn in _LOG_HOOKS:
        _ip.events.register(ev, fn)
except NameError:
    pass                                                # not inside a notebook

log(f"Run started | notebook {NOTEBOOK_VERSION} | Python {sys.version.split()[0]} | {resources()}")
log("Settings: " + ", ".join(f"{k}={globals()[k]!r}" for k in
    ["PROJECT", "VOICE", "SPEED_SCALE", "WER_RETRY", "ASR_MODEL", "CLIP_STEPS", "MAX_SLOWMO", "SMOOTH_MOTION", "SHARPEN", "CONTINUITY_CHECK", "CONTINUITY_RETRIES", "FRAMES_ON_DEMAND",
     "MUSIC_STYLE", "BURN_SUBS", "VIDEO_SPEED", "CACHE_THRESHOLD", "MODEL_CACHE_ON_DRIVE", "MUSIC_GAIN", "SFX_GAIN_DB", "FPS", "W", "H", "X264_PRESET"]))
print("Logging to:", LOG.path, "(moves to Google Drive in step 3)")
