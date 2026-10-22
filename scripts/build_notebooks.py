"""Builds the Kaggle notebooks, one per model, each embedding the nwrd_public/ package and the
data manifest so it runs without cloning the repository:

    notebooks/canet_b4.ipynb            CANet-B4 (ours)
    notebooks/unet.ipynb                the NWRD paper's UNet architecture
    notebooks/deeplabv3plus_r50.ipynb   DeepLabV3+ R50

    python scripts/build_notebooks.py

The three are identical except for their `MODEL` line; the scale (`PRESET`) is set on the copy
scripts/run_on_kaggle.py pushes. Re-run after changing anything in nwrd_public/ or the manifest;
tests/test_notebooks.py fails while a notebook is stale.
"""

import json
import os

from run_on_kaggle import BUDGET_LINE, DATASET, EXPECT_OFF, MODELS, PRESET_LINE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = [
    '__init__', 'config', 'dataset', 'windows', 'models', 'metrics', 'train', 'evaluate',
    'compare', 'run',
]  # fmt: skip

INTRO = f"""# Wheat-rust segmentation on the public NWRD release: one model, one scale

Trains the model named in `MODEL` (next cell) at the scale named in `PRESET` under the protocol shared by every model of the study: CANet-B4 (ours), the NWRD paper's UNet architecture and DeepLabV3+ R50. Three seeds; the checkpoint and the threshold are chosen on the 9 validation photographs; the 10 test photographs are scored once, every pixel at native resolution. The three notebooks (`canet_b4`, `unet`, `deeplabv3plus_r50`) are identical except for `MODEL`, and `scripts/run_on_kaggle.py --preset 128px` sets `PRESET` on the copy it pushes.

**The data.** The public NWRD release (Anwar and Shafait, Zenodo doi:10.5281/zenodo.21672928, CC BY 4.0), attached as the Kaggle dataset `{DATASET}`. Before anything else, each of its 200 files is checked against its MD5 in `data/manifest.json`, taken from the Zenodo archive, and the run stops on any difference. Windows are cut on a stride fitted to each photograph's size, without padding: 512px windows at native resolution, or 128px windows of the photograph halved with Lanczos (the paper's scale). The training windows go to local disk and are cut again, identically, in every session.

**Kaggle setup**
1. *Add Input* → dataset **`{DATASET}`**.
2. *Settings* → Accelerator **GPU T4** (one GPU is used), **Internet on** (pip and ImageNet weights).
3. *Save Version* → **Save & Run All (Commit)**. A session runs for up to 12 h.

**If a session stops at its time budget** (the log ends in `INCOMPLETE`), continue it with `python scripts/run_on_kaggle.py --model <model> [--preset 128px] --resume --watch` from the repository, or add this notebook's latest version as an input and commit again. Repeat until the log has no `INCOMPLETE`.

Estimated T4 time for three seeds of 30 epochs: at 512px CANet-B4 3.0 h, UNet 6.2 h, DeepLabV3+ R50 2.0 h; at 128px 2.6, 1.5 and 1.0 h. Each fits in one session."""

SETUP = """import glob, json, os, shutil, subprocess, sys
!pip install -q segmentation-models-pytorch==0.5.0 efficientnet_pytorch==0.7.1 "albumentations>=2.0,<3"
import torch
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'enable a GPU accelerator'
os.makedirs('code/nwrd_public', exist_ok=True)
os.makedirs('code/data', exist_ok=True)"""

CONFIG = f"""# ── Run settings ────────────────────────────────────────────────────────────────
MODEL = '{{model}}'   # canet_b4 (ours), unet or deeplabv3plus_r50
{PRESET_LINE}   # 512px (native resolution) or 128px (the paper's scale): nwrd_public/config.py
NUM_WORKERS = 4
# Kaggle kills a session at 12 h. Lower this when the weekly GPU quota has less than ~12 h left,
# so the session stops cleanly (and can be resumed) instead of being killed.
# scripts/run_on_kaggle.py --budget-hours sets it on the pushed copy.
{BUDGET_LINE}
# True = this session must continue earlier runs; stop at once if none are attached rather than
# silently retraining everything. scripts/run_on_kaggle.py --resume sets it on the pushed copy.
{EXPECT_OFF}"""

STORAGE = """# Runs go to /kaggle/working/runs, which Kaggle keeps with each committed version.
OUT = '/kaggle/working/runs' if os.path.isdir('/kaggle/working') else os.path.abspath('runs')
os.makedirs(OUT, exist_ok=True)
print(MODEL, PRESET, '| runs ->', OUT, '| time budget', KAGGLE_TIME_BUDGET_HOURS, 'h')"""

RESTORE = """# Continue an earlier session. Kaggle offers the output of a finished version as an input
# (not of one killed at the session limit); its runs/ is copied into place here, once it is
# checked to be this notebook's model and scale.
prev = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/runs/config.json', recursive=True))
for d in prev:
    found = json.load(open(os.path.join(d, 'config.json')))
    if (found.get('model'), found.get('preset')) != (MODEL, PRESET):
        raise RuntimeError("the attached %s holds runs of %s %s, not %s %s: attach this kernel's own output"
                           % (d, found.get('model'), found.get('preset'), MODEL, PRESET))
if prev and not os.path.exists(os.path.join(OUT, 'config.json')):
    shutil.copytree(prev[0], OUT, dirs_exist_ok=True)
    print('restored previous runs from', prev[0])
    with open(os.path.join(OUT, 'log.txt'), 'a', encoding='utf-8') as f:
        f.write('--- new session: restored previous runs from %s\\n' % prev[0])
states = {}
for d in sorted(glob.glob(os.path.join(OUT, '*', 'seed*'))):
    states[d] = ('done' if os.path.exists(os.path.join(d, 'metrics.json')) else
                 'trained' if os.path.exists(os.path.join(d, 'TRAINED')) else
                 'partial' if os.path.exists(os.path.join(d, 'last.pt')) else 'empty')
    print(' ', os.path.relpath(d, OUT), states[d])
print('%d run directories found' % len(states))
if EXPECT_RESTORE and not any(s != 'empty' for s in states.values()):
    print('/kaggle/input:', sorted(glob.glob('/kaggle/input/*/*/*'))[:30])
    raise RuntimeError('Resume was requested but no earlier runs/ was found under /kaggle/input. '
                       'Stopping now instead of retraining every run from scratch.')"""

RUN = """# The release is found under /kaggle/input and checked against code/data/manifest.json first.
cmd = [sys.executable, '-m', 'nwrd_public.run', '--model', MODEL, '--preset', PRESET, '--out', OUT,
       '--time-budget-hours', str(KAGGLE_TIME_BUDGET_HOURS), '--num-workers', str(NUM_WORKERS)]
print(' '.join(cmd))
proc = subprocess.Popen(cmd, cwd='code', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end='')
code = proc.wait()
print('exit code', code, '(3 = stopped at the time budget: continue with scripts/run_on_kaggle.py --model',
      MODEL, '--preset', PRESET, "--resume, or attach this version's output as an input and commit again)")
# Fail the version, so Kaggle reports ERROR instead of a COMPLETE run that trained nothing.
if code not in (0, 3):
    raise RuntimeError('the pipeline failed (exit code %d); the traceback is in runs/log.txt' % code)"""

SHOW = """from IPython.display import Markdown, display
summary = os.path.join(OUT, MODEL, 'summary.md')
if os.path.exists(summary):
    display(Markdown(open(summary, encoding='utf-8').read()))"""

OUTRO = """## Taking the results back to the repository

`python scripts/run_on_kaggle.py --model <model> [--preset 128px] --status` downloads this version's output to `runs_kaggle/<scale>/<model>/<account>_v<N>/` (weights stay on Kaggle). Then `python scripts/collect_results.py runs_kaggle/<scale>/<model>/<account>_v<N>` copies its runs into `results/<scale>/<model>/` and rebuilds the side-by-side comparison of the models collected so far: `results/<scale>/comparison.md` and `results/summary.csv`."""


def md(src: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.splitlines(True)}


def code(src: str) -> dict:
    return {
        'cell_type': 'code',
        'metadata': {},
        'execution_count': None,
        'outputs': [],
        'source': src.splitlines(True),
    }


def build(model: str) -> dict:
    """The notebook of `model` as a JSON-ready dict."""
    cells = [
        md(INTRO),
        code(SETUP),
        code(CONFIG.format(model=model)),
        code(STORAGE),
        md('### Pipeline source (generated by `scripts/build_notebooks.py`; edit the repository)'),
    ]
    with open(os.path.join(ROOT, 'data', 'manifest.json'), encoding='utf-8') as f:
        cells.append(code('%%writefile code/data/manifest.json\n' + f.read()))
    for m in MODULES:
        with open(os.path.join(ROOT, 'nwrd_public', f'{m}.py'), encoding='utf-8') as f:
            cells.append(code(f'%%writefile code/nwrd_public/{m}.py\n' + f.read()))
    cells += [
        md('### Resume'),
        code(RESTORE),
        md('### Run'),
        code(RUN),
        code(SHOW),
        md(OUTRO),
    ]
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    return {
        'cells': cells,
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'},
            'accelerator': 'GPU',
            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True},
        },
    }


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + '\n'


def main() -> None:
    for model in MODELS:
        out = os.path.join(ROOT, 'notebooks', f'{model}.ipynb')
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8', newline='\n') as f:
            f.write(render(build(model)))
        print('wrote', out)


if __name__ == '__main__':
    main()
