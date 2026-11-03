"""Lightweight startup diagnostics: no torch, NumPy, Diffusers or CUDA imports."""
from pathlib import Path
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import uuid


def memory_snapshot():
    result = {}
    try:
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        for key in ('MemTotal', 'MemAvailable'):
            result[key + '_bytes'] = int(fields[key].split()[0]) * 1024
    except (OSError, ValueError, KeyError):
        pass
    for name in ('memory.current', 'memory.max', 'memory.events'):
        try:
            value = Path('/sys/fs/cgroup', name).read_text().strip()
            result[name] = int(value) if value.isdigit() else value
        except OSError:
            pass
    return result


class RuntimeDiagnostics:
    def __init__(self, directory, interval=5):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = directory / f'runtime-{stamp}-{uuid.uuid4().hex[:8]}.jsonl'
        self.stage = 'Drive mounted'
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.interval = interval
        self.note('Drive mounted')
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _write(self):
        record = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      stage=self.stage, pid=os.getpid(), python=platform.python_version(),
                      memory=memory_snapshot())
        with self.path.open('a') as handle:
            handle.write(json.dumps(record) + '\n')
            handle.flush()
            os.fsync(handle.fileno())

    def note(self, stage):
        with self.lock:
            self.stage = str(stage)
            self._write()
        print(stage, flush=True)

    def _sample(self):
        while not self.stopped.wait(self.interval):
            try:
                with self.lock:
                    self._write()
            except OSError:
                # A detached Drive should not itself terminate the kernel.
                return

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=2)


def installed_versions():
    names = ('torch', 'torchvision', 'torchaudio', 'numpy', 'diffusers',
             'transformers', 'accelerate', 'bitsandbytes', 'torchao')
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_constraints(path):
    # Preserve Colab's installed ABI-coupled packages and CUDA distribution versions.
    versions = installed_versions()
    pins = {k: versions[k] for k in ('torch', 'torchvision', 'torchaudio', 'numpy') if versions[k]}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get('Name', '')
        if name.lower().startswith('nvidia-') or name.lower() == 'triton':
            pins[name] = dist.version
    Path(path).write_text(''.join(f'{name}=={version}\n' for name, version in sorted(pins.items())))
    return pins


def lightweight_preflight(references, output):
    # Inspect hardware with nvidia-smi, not by loading torch or the pipeline.
    import hashlib
    from PIL import Image
    refs = {}
    for cid, path in references.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Upload the {cid} reference to {path}')
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            size = list(im.size)
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        refs[cid] = dict(file=str(path), sha256=digest, size=size)
    report = dict(python=platform.python_version(), packages=installed_versions(),
                  memory=memory_snapshot(), references=refs, gpu=None,
                  disk_free_gb=round(shutil.disk_usage(Path(output).parent).free / 2**30, 2))
    if shutil.which('nvidia-smi'):
        check = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=20)
        if check.returncode == 0:
            report['gpu'] = check.stdout.strip() or None
        else:
            report['gpu_probe_error'] = check.stderr[-1000:]
    Path(output).write_text(json.dumps(report, indent=2))
    return report
