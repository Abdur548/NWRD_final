"""Runs one model's notebook on a Kaggle GPU from the command line.

Needs a Kaggle API token once: kaggle.com -> Settings -> API -> Create New Token, saved as
~/.kaggle/kaggle.json (or in the folder KAGGLE_CONFIG_DIR names).

    python scripts/run_on_kaggle.py --model canet_b4 --watch               # 512px: push, poll
    python scripts/run_on_kaggle.py --model unet --preset 128px --watch    # the paper's scale
    python scripts/run_on_kaggle.py --model canet_b4 --resume --watch      # continue a stopped run
    python scripts/run_on_kaggle.py --model canet_b4 --status              # check, download output
    python scripts/run_on_kaggle.py --model unet --budget-hours 6 --watch  # little quota left

Each (model, scale) is its own private kernel, GPU-enabled, with internet on and the public
release `bcap52/nwrd-wheat` attached; the notebook checks every file of it against
data/manifest.json before training. `--resume` attaches the kernel's previous output, which the
notebook's restore cell copies into place, so training continues instead of restarting; if
nothing can be restored, the notebook stops at once instead of retraining from scratch. Kaggle
only attaches the output of a version that finished (complete or error), so a version killed at
the 12 h limit, or when the weekly GPU quota runs out, cannot be resumed from: `--budget-hours`
stops a session cleanly before that. Repeat until the log has no INCOMPLETE line.

Output lands in runs_kaggle/<scale>/<model>/<account>_v<version>/, one folder per pushed version,
so sessions and accounts never overwrite each other; scripts/collect_results.py takes it into
results/. Weights stay on Kaggle unless --with-checkpoints.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = 'bcap52/nwrd-wheat'
# nwrd_public.config's COMPARED_MODELS, then its ABLATION_MODELS (run at 512px only).
MODELS = (
    'canet_b4',
    'unet',
    'deeplabv3plus_r50',
    'deeplabv3plus_r50_scratch',
    'unet_r50',
    'deeplabv3plus_b4',
)
PRESETS = ('512px', '128px')
PUSH_ROOT = os.path.join(ROOT, 'build', 'kaggle_push')
OUT_ROOT = os.path.join(ROOT, 'runs_kaggle')
# Lines of the notebook the pushed copy rewrites; each occurs exactly once (tests/test_notebooks).
PRESET_LINE = "PRESET = '512px'"
EXPECT_OFF, EXPECT_ON = 'EXPECT_RESTORE = False', 'EXPECT_RESTORE = True'
BUDGET_LINE = 'KAGGLE_TIME_BUDGET_HOURS = 11.0'
NO_WEIGHTS = r'^(?!.*\.pth?$).*$'  # neither checkpoints (*.pt) nor exported models (*.pth)
STATES = {
    'COMPLETE': 'complete',
    'ERROR': 'error',
    'RUNNING': 'running',
    'QUEUED': 'queued',
    'CANCEL_REQUESTED': 'cancelled',
    'CANCEL_ACKNOWLEDGED': 'cancelled',
}
STOPPED = ('complete', 'error', 'cancelled')  # a stopped run's output is still worth fetching


def slugify(title: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def kernel(model: str, preset: str) -> dict:
    """Kaggle derives a kernel's slug from its title, so the title decides where it lives:
    renaming one moves the kernel, and --resume / --status would look in the wrong place."""
    title = f'NWRD public {model} {preset}'
    return {
        'model': model,
        'preset': preset,
        'title': title,
        'slug': slugify(title),
        'notebook': os.path.join(ROOT, 'notebooks', f'{model}.ipynb'),
        'push_dir': os.path.join(PUSH_ROOT, f'{model}_{preset}'),
    }


def flags(k: dict) -> str:
    return f' --model {k["model"]}' + (f' --preset {k["preset"]}' if k['preset'] != '512px' else '')


def version_file(k: dict, user: str) -> str:
    return os.path.join(k['push_dir'], f'last_pushed_version_{user}.txt')


def output_dir(k: dict, user: str, version) -> str:
    """Each account numbers its kernel versions from 1, so folders carry the account name."""
    return os.path.join(OUT_ROOT, k['preset'], k['model'], f'{user}_v{version}')


def kaggle(*args: str, check: bool = True) -> str:
    """Runs the kaggle CLI through the current interpreter, so no PATH setup is needed.

    In UTF-8 mode: the CLI saves the kernel log with the locale's encoding, and on Windows
    (cp1252) the log's non-ASCII characters made that fail, leaving an empty file."""
    p = subprocess.run(
        [sys.executable, '-m', 'kaggle', *args],
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        env={**os.environ, 'PYTHONUTF8': '1'},
    )
    out = (p.stdout or '') + (p.stderr or '')
    if check and p.returncode != 0:
        raise SystemExit(f'kaggle {" ".join(args)} failed:\n{out.strip()}')
    return out.strip()


def username() -> str:
    """The account the kaggle CLI will use: KAGGLE_CONFIG_DIR wins over ~/.kaggle, as in the CLI."""
    config_dir = os.environ.get('KAGGLE_CONFIG_DIR')
    for path in (
        os.path.join(config_dir, 'kaggle.json') if config_dir else None,
        os.path.expanduser('~/.kaggle/kaggle.json'),
    ):
        if path and os.path.exists(path):
            with open(path) as f:
                return json.load(f)['username']
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    raise SystemExit(
        'No Kaggle credentials. Save kaggle.json to ~/.kaggle/ '
        '(kaggle.com -> Settings -> API -> Create New Token).'
    )


def notebook_source(
    notebook: str, preset: str = '512px', resume: bool = False, budget_hours: float | None = None
) -> str:
    """The notebook as pushed: at `preset`'s scale; with resume, the restore cell must find
    earlier runs or stop; with budget_hours, the session stops cleanly after that many hours."""
    with open(notebook, encoding='utf-8') as f:
        text = f.read()
    for line in (PRESET_LINE, EXPECT_OFF, BUDGET_LINE):
        if text.count(line) != 1:
            raise SystemExit(
                f'expected one "{line}" line in {notebook}; rebuild it with '
                'scripts/build_notebooks.py'
            )
    text = text.replace(PRESET_LINE, f"PRESET = '{preset}'")
    if resume:
        text = text.replace(EXPECT_OFF, EXPECT_ON)
    if budget_hours is not None:
        text = text.replace(BUDGET_LINE, f'KAGGLE_TIME_BUDGET_HOURS = {float(budget_hours)}')
    return text


def kernel_metadata(k: dict, kernel_id: str, resume: bool) -> dict:
    return {
        'id': kernel_id,
        'title': k['title'],
        'code_file': os.path.basename(k['notebook']),
        'language': 'python',
        'kernel_type': 'notebook',
        'is_private': True,
        'enable_gpu': True,
        'enable_internet': True,
        'dataset_sources': [DATASET],
        'competition_sources': [],
        # Attaching the kernel's own previous output lets the notebook resume.
        'kernel_sources': [kernel_id] if resume else [],
    }


def push(k: dict, user: str, resume: bool, budget_hours: float | None = None) -> None:
    kernel_id = f'{user}/{k["slug"]}'
    os.makedirs(k['push_dir'], exist_ok=True)
    with open(os.path.join(k['push_dir'], 'kernel-metadata.json'), 'w') as f:
        json.dump(kernel_metadata(k, kernel_id, resume), f, indent=2)
    pushed = os.path.join(k['push_dir'], os.path.basename(k['notebook']))
    with open(pushed, 'w', encoding='utf-8') as f:
        f.write(notebook_source(k['notebook'], k['preset'], resume, budget_hours))
    out = kaggle('kernels', 'push', '-p', k['push_dir'])
    print(out)
    # The CLI exits 0 when Kaggle refuses a push ("Kernel push error: Maximum batch GPU session
    # count of 2 reached."), so only a version number in the reply means a session started.
    m = re.search(r'version (\d+)', out)
    if not m:
        raise SystemExit(f'Kaggle did not start {k["title"]}: {out}')
    with open(version_file(k, user), 'w') as f:
        f.write(m.group(1))


def parse_status(text: str) -> str:
    """Maps `kaggle kernels status` output to complete / error / running / queued /
    cancelled, or unknown (for instance when the CLI itself failed)."""
    m = re.search(r'KernelWorkerStatus\.([A-Z_]+)', text) or re.search(
        r'has status "([A-Z_]+)"', text
    )
    return STATES.get(m.group(1), 'unknown') if m else 'unknown'


def status(kernel_id: str) -> tuple[str, str]:
    text = kaggle('kernels', 'status', kernel_id, check=False)
    return parse_status(text), text


def fetch_output(k: dict, user: str, pattern: str | None = NO_WEIGHTS) -> str:
    """Downloads the latest version's output into its own folder and prints the model's summary
    and the log tail. The CLI always serves the latest version, so the folder is named after the
    last version this script pushed from this account."""
    vfile = version_file(k, user)
    version = 'latest'
    if os.path.exists(vfile):
        with open(vfile) as f:
            version = f.read().strip()
    out_dir = output_dir(k, user, version)
    os.makedirs(out_dir, exist_ok=True)
    args = ['kernels', 'output', f'{user}/{k["slug"]}', '-p', out_dir, '-q']
    if pattern:
        args += ['--file-pattern', pattern]
    print(kaggle(*args, check=False) or f'downloaded to {out_dir}')
    summary = os.path.join(out_dir, 'runs', k['model'], 'summary.md')
    if os.path.exists(summary):
        with open(summary, encoding='utf-8') as f:
            print('\n' + f.read())
    else:
        print(f'no {k["model"]}/summary.md in {out_dir}')
    log = os.path.join(out_dir, 'runs', 'log.txt')
    if os.path.exists(log):
        with open(log, encoding='utf-8') as f:
            tail = f.read().splitlines()[-15:]
        print('--- log tail ---\n' + '\n'.join(tail))
        if any('INCOMPLETE' in line for line in tail):
            print(f'\nStopped at the time budget: re-run with{flags(k)} --resume --watch.')
        else:
            print(f'\nTo results/: python scripts/collect_results.py {out_dir}')
    return out_dir


def main() -> None:
    sys.stdout.reconfigure(encoding='utf-8')  # summaries have ±; Windows consoles are cp1252
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--preset', choices=PRESETS, default='512px')
    ap.add_argument('--resume', action='store_true', help="attach the kernel's previous output")
    ap.add_argument('--watch', action='store_true', help='poll until the run finishes')
    ap.add_argument('--status', action='store_true', help='report status only, do not push')
    ap.add_argument('--poll-minutes', type=float, default=10)
    ap.add_argument(
        '--budget-hours',
        type=float,
        default=None,
        help='stop training cleanly after this many hours, e.g. when the weekly GPU quota has '
        'less than a full session left',
    )
    ap.add_argument('--with-checkpoints', action='store_true', help='download every file')
    a = ap.parse_args()

    k, user = kernel(a.model, a.preset), username()
    kernel_id = f'{user}/{k["slug"]}'
    pattern = None if a.with_checkpoints else NO_WEIGHTS
    if a.status:
        state, text = status(kernel_id)
        print(text)
        if state in STOPPED:
            fetch_output(k, user, pattern)
        return

    push(k, user, a.resume, a.budget_hours)
    print(f'https://www.kaggle.com/code/{kernel_id}')
    if not a.watch:
        print(f'Check later:  python scripts/run_on_kaggle.py{flags(k)} --status')
        return
    while True:
        time.sleep(a.poll_minutes * 60)
        state, text = status(kernel_id)
        print(f'{time.strftime("%H:%M")} {state}')
        if state in STOPPED:
            print(text)
            fetch_output(k, user, pattern)
            return


if __name__ == '__main__':
    main()
