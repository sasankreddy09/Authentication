"""Rebuild only the cinematic Telugu 'How 3D works' notebook from editable sources."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'targets/three_d_telugu'
OUTPUT = HERE / 'notebooks/colab_three_d_friends_te_v3_storybook.ipynb'
ORDER = ['settings', 'preflight', 'log', 'install', 'setup', 'lib', 'budget', 'plan', 'helpers',
         'prefetch', 'narration', 'timeline', 'diagrams', 'keyframes', 'editlib', 'logic', 'bench',
         'clips', 'music', 'sfx', 'render', 'qc', 'metadata', 'preview']


def build():
    modules = {p.name: p.read_text() for p in sorted((HERE / 'engine/lib').glob('*.py'))
               if p.name not in ('studio.py', 'studio_run.py')}
    plan = json.loads((HERE / 'plans/three_d_friends_te_v3_storybook_plan.json').read_text())
    cells = []
    def add(kind, text, tag):
        cell = {'cell_type': kind, 'id': hashlib.sha256(tag.encode()).hexdigest()[:12],
                'metadata': {'tags': [tag]}, 'source': text.splitlines(True)}
        if kind == 'code':
            compile(text, f'Telugu {tag}', 'exec')
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add('markdown', (TARGET / 'README.md').read_text(), 'intro')
    for tag in ORDER:
        src = (TARGET / 'cells' / f'{tag}.py').read_text()
        src = src.replace('__LIB_FILES__', json.dumps(modules, ensure_ascii=True))
        src = src.replace('__PLAN_JSON__', json.dumps(plan, indent=1, ensure_ascii=False))
        if tag in ORDER[ORDER.index('plan'):]:
            src = f'BUDGET.check({tag!r})\n' + src
        add('code', src, tag)
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'colab': {'provenance': [], 'gpuType': 'T4'}, 'accelerator': 'GPU',
                         'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                         'language_info': {'name': 'python'}}, 'cells': cells}


def serialized():
    return json.dumps(build(), indent=1, ensure_ascii=False) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check', action='store_true', help='fail if committed notebook differs from its sources')
    args = ap.parse_args()
    result = serialized()
    if args.check:
        if OUTPUT.read_text() != result:
            raise SystemExit('Telugu notebook is stale: run python colab/build_telugu.py')
        print('Telugu notebook matches its sources.')
    else:
        OUTPUT.write_text(result)
        print('Wrote', OUTPUT)
