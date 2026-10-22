"""Copies a Kaggle kernel's finished runs into results/, without weights, then rebuilds that
scale's side-by-side comparison and results/summary.csv.

    python scripts/collect_results.py runs_kaggle/512px/canet_b4/abdur548_v1

The kernel's runs/config.json says which model and scale it ran. Its files land in
results/<scale>/<model>/: config.json, log.txt and windows.json, and per seed metrics.json,
history.json, val_threshold_sweep.json, test_counts.npz and paper_patches.npz. The model's earlier
files at that scale are replaced, so a results folder never mixes two versions of a kernel. A run
without metrics.json (unfinished) stops the copy unless --partial.
"""

import argparse
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a checkout without installing the package

from nwrd_public.compare import compare, write_summary_csv  # noqa: E402
from nwrd_public.config import preset_of  # noqa: E402

TOP_FILES = ('config.json', 'log.txt', 'windows.json')
RUN_FILES = (
    'metrics.json',
    'history.json',
    'val_threshold_sweep.json',
    'test_counts.npz',
    'paper_patches.npz',
)


def collect(src: str, results_root: str, preset: str | None = None, partial: bool = False) -> str:
    """Copies the runs under `src` (a kernel's downloaded output) into
    <results_root>/<preset>/<model>/ and returns that folder. The preset is read from the runs'
    window size and scale unless given."""
    configs = glob.glob(os.path.join(glob.escape(src), '**', 'config.json'), recursive=True)
    if len(configs) != 1:
        raise SystemExit(f'expected one runs/config.json under {src}, found {configs}')
    runs = os.path.dirname(configs[0])
    with open(configs[0], encoding='utf-8') as f:
        cfg = json.load(f)
    model, preset = cfg['model'], preset or preset_of(cfg['window'], cfg['scale'])
    seed_dirs = sorted(glob.glob(os.path.join(glob.escape(runs), model, 'seed*')))
    finished = [d for d in seed_dirs if os.path.exists(os.path.join(d, 'metrics.json'))]
    if not finished or (len(finished) < len(seed_dirs) and not partial):
        raise SystemExit(
            f'{len(finished)} of {len(seed_dirs)} runs in {runs} are finished; resume the kernel '
            'or pass --partial'
        )
    for d in finished:  # each run's own record must agree with the kernel's settings
        with open(os.path.join(d, 'metrics.json'), encoding='utf-8') as f:
            m = json.load(f)
        if (m['model'], m['window'], m['scale']) != (model, cfg['window'], cfg['scale']):
            raise SystemExit(
                f'{d} was scored as {m["model"]} at {m["window"]}px x{m["scale"]:g}, but its '
                f'kernel ran {model} at {cfg["window"]}px x{cfg["scale"]:g}: not collecting'
            )

    dst = os.path.join(results_root, preset, model)
    if os.path.isdir(dst):
        print('replacing', dst)
        shutil.rmtree(dst)
    os.makedirs(dst)
    for name in TOP_FILES:
        if os.path.exists(os.path.join(runs, name)):
            shutil.copy2(os.path.join(runs, name), dst)
    for d in finished:
        out = os.path.join(dst, os.path.basename(d))
        os.makedirs(out)
        for name in RUN_FILES:
            if os.path.exists(os.path.join(d, name)):
                shutil.copy2(os.path.join(d, name), out)
    print(f'copied {len(finished)} runs of {model} at {preset} to {dst}')
    compare(os.path.join(results_root, preset), log=lambda _msg: None)
    print(
        'rebuilt',
        os.path.join(results_root, preset, 'comparison.md'),
        'and',
        write_summary_csv(results_root),
    )
    return dst


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('src', help="a kernel's downloaded output (runs_kaggle/<scale>/<model>/...)")
    ap.add_argument('--results', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--partial', action='store_true', help='copy the finished runs only')
    a = ap.parse_args(argv)
    collect(a.src, a.results, partial=a.partial)


if __name__ == '__main__':
    main()
