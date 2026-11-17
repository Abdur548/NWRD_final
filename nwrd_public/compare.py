"""The side-by-side log of the three models at one scale, and each model's own summary.

Outputs in a scale's folder (results/512px or results/128px): comparison.md, comparison.json,
figures/training_curves.png and <model>/summary.md; and results/summary.csv over both scales.
Every number is computed from the files each run saves (metrics.json, history.json and the
per-photograph counts in test_counts.npz), so the reports are rebuilt without checkpoints or
photographs. Each run's paper_patches.npz keeps the per-patch counts behind its paper-patch
scores, for recomputing other averages offline:

    python -m nwrd_public.compare results
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

from .config import COMPARED_MODELS
from .metrics import dice, iou
from .models import MODELS
from .windows import grid, scaled_shape

METRICS = ('IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'Accuracy')
OURS = 'canet_b4'
SCALES = ('512px', '128px')

# Published NWRD paper results (Anwar et al., Sensors 2023, doi:10.3390/s23156942) for UNet + APF
# on the release's 10 test photographs. Table 3: the 2,737 patches the paper's test.py scores
# (128px patches of the half-size photographs), averaged per batch as that code does. Table 4: a
# separately trained UNet (stride 128, background-only patches removed from training and test,
# paper §5), on the test patches with rust; its IoU was not computed by test.py.
PAPER_TEST_PATCHES = 2737
PAPER = [
    {
        'table': 'Table 3',
        'patches': 'all test patches',
        'kind': 'authors',
        'scores': {'Precision': 0.506, 'Recall': 0.624, 'F1': 0.557},
    },
    {
        'table': 'Table 4',
        'patches': 'background-only patches removed',
        'kind': 'authors_rust_only',
        'scores': {'Precision': 0.593, 'Recall': 0.552, 'F1': 0.564, 'IoU': 0.438},
    },
]
PAPER_ROWS = (  # (threshold key, kind, label) of our runs on the paper's patches
    ('@0.5', 'authors', "all patches, the authors' averaging, threshold 0.5"),
    ('@0.5', 'authors_rust_only', "rust patches (≈ Table 4), the authors' averaging, 0.5"),
    ('@thr', 'pooled', 'all patches, pooled, validation threshold'),
    ('@thr', 'rust_only', 'rust patches, pooled, validation threshold'),
)
# The two common sizes of the release, for the time per photograph: (height, width) in pixels.
PHOTO_SIZES = {
    'Nikon D3200, 6016×4000': (4000, 6016),
    'Huawei PAR-LX1M, 4608×3456 or 3456×4608': (3456, 4608),
}

VERDICTS = {
    'improves': 'improves on',
    'comparable': 'is comparable to',
    'worse': 'is worse than',
}


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_runs(scale_dir: str) -> dict:
    """{model: {seed: run}} for every run with a metrics.json. A run holds its directory and
    metrics, plus 'history' (the whole history.json), 'photos', 'cameras' and 'counts' (the
    test photographs' confusion counts at the run's threshold) and 'paper' (the path of its
    paper_patches.npz) where those files exist."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(scale_dir, '*', 'seed*', 'metrics.json'))):
        run_dir = os.path.dirname(path)
        model = os.path.basename(os.path.dirname(run_dir))
        seed = int(os.path.basename(run_dir)[len('seed') :])
        run = {'dir': run_dir, 'metrics': _read_json(path)}
        history = _read_json(os.path.join(run_dir, 'history.json'))
        if history is not None:
            run['history'] = history
        counts = os.path.join(run_dir, 'test_counts.npz')
        if os.path.exists(counts):
            with np.load(counts) as c:
                run['photos'], run['cameras'] = c['photos'], c['cameras']
                run['counts'] = c['at_thr'].astype(np.int64)
        if os.path.exists(os.path.join(run_dir, 'paper_patches.npz')):
            run['paper'] = os.path.join(run_dir, 'paper_patches.npz')
        runs.setdefault(model, {})[seed] = run
    return runs


def paired_bootstrap(
    ours: dict, base: dict, seeds: list, n_boot: int = 5000, rng_seed: int = 0
) -> dict:
    """Paired bootstrap of the seed-averaged difference in pooled test IoU and Dice. Each
    resample draws test photographs with replacement, the same draws for both models and every
    seed: pixels of one photograph share plant, light and annotator, so the photograph is the
    unit of resampling."""
    for s in seeds:
        if not np.array_equal(ours[s]['photos'], base[s]['photos']):
            raise ValueError(f'seed {s}: the two models were tested on different photographs')
    n = len(ours[seeds[0]]['photos'])
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    w = np.stack([np.bincount(d, minlength=n) for d in draws])
    out = {}
    for name, fn in (('IoU', iou), ('Dice', dice)):
        d = np.mean([fn(w @ ours[s]['counts']) - fn(w @ base[s]['counts']) for s in seeds], 0)
        point = np.mean(
            [fn(ours[s]['counts'].sum(0)) - fn(base[s]['counts'].sum(0)) for s in seeds]
        )
        out[name] = {
            'delta': float(point),
            'ci95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'p_two_sided': float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))),
            'units': int(n),
        }
    return out


def seed_ttest(ours: dict, base: dict, seeds: list) -> dict | None:
    """Paired t-test over seeds of test IoU; it captures training-run variance."""
    if len(seeds) < 2:
        return None
    from scipy import stats

    a = [ours[s]['metrics']['test@thr']['IoU'] for s in seeds]
    b = [base[s]['metrics']['test@thr']['IoU'] for s in seeds]
    t, p = stats.ttest_rel(a, b)
    return {'t': float(t), 'p': float(p), 'n_seeds': len(seeds)}


def seed_wins(ours: dict, base: dict, seeds: list) -> int:
    """Number of seeds on which `ours` has the higher test IoU."""
    return sum(
        ours[s]['metrics']['test@thr']['IoU'] > base[s]['metrics']['test@thr']['IoU'] for s in seeds
    )


def verdict(boot_photo_iou: dict, wins: int) -> str:
    """The decision rule, fixed before any result existed (docs/protocol.md §5).

    'improves': higher mean test IoU, the 95% CI of the IoU difference that resamples test
    photographs excludes 0, and CANet wins on at least 2 of 3 seeds. 'worse': that CI lies
    entirely below 0. 'comparable': otherwise. The rule assumes all three seeds; with fewer,
    the reading is interim."""
    lo, hi = boot_photo_iou['ci95']
    if boot_photo_iou['delta'] > 0 and lo > 0 and wins >= 2:
        return 'improves'
    if hi < 0:
        return 'worse'
    return 'comparable'


def significance(runs: dict) -> dict:
    """CANet-B4 against each pre-registered baseline, on the seeds both have. The ablation's
    models (config.ABLATION_MODELS) are exploratory and never judged by the decision rule."""
    tests = {}
    if OURS not in runs:
        return tests
    for base in (m for m in runs if m != OURS and m in COMPARED_MODELS):
        seeds = sorted(set(runs[OURS]) & set(runs[base]))
        if not seeds:
            continue
        boot = paired_bootstrap(runs[OURS], runs[base], seeds)
        wins = seed_wins(runs[OURS], runs[base], seeds)
        tests[base] = {
            'seeds': seeds,
            'bootstrap_photo': boot,
            'seed_ttest': seed_ttest(runs[OURS], runs[base], seeds),
            'canet_wins': wins,
            'verdict': verdict(boot['IoU'], wins),
        }
    return tests


def mean_std(vals) -> tuple[float, float]:
    """Mean and sample standard deviation (0 for a single value)."""
    vals = np.asarray(vals, float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def _fmt(m: float, s: float, n: int) -> str:
    return f'{m:.4f} ± {s:.4f}' if n > 1 else f'{m:.4f}'


def summarise(runs: dict) -> dict:
    summary = {}
    for model, seeds in runs.items():
        ms = [seeds[s]['metrics'] for s in sorted(seeds)]
        hs = [seeds[s]['history']['history'] for s in sorted(seeds)]
        row = {'seeds': sorted(seeds), 'n': len(ms)}
        for split in ('test@thr', 'test@0.5', 'val@thr'):
            row[split] = {k: mean_std([m[split][k] for m in ms]) for k in METRICS}
        row['threshold'] = mean_std([m['threshold'] for m in ms])
        row['threshold_at_grid_edge'] = any(m['threshold_at_grid_edge'] for m in ms)
        row['best_epoch'] = [m['best_epoch'] for m in ms]
        row['train_min_per_epoch'] = mean_std([np.mean(h['epoch_seconds']) / 60 for h in hs])
        row['paper'] = {
            thr: {
                kind: {
                    k: mean_std([m['paper_patches'][thr][kind][k] for m in ms])
                    for k in ms[0]['paper_patches'][thr][kind]
                }
                for kind in ms[0]['paper_patches'][thr]
            }
            for thr in ('@thr', '@0.5')
        }
        summary[model] = row
    return summary


def _order(models) -> list[str]:
    return [m for m in (OURS,) if m in models] + sorted(m for m in models if m != OURS)


def _name(model: str) -> str:
    return f'**{model}** (ours)' if model == OURS else model


def _photo_table(runs: dict, order: list[str]) -> list[str]:
    """IoU of each test photograph at each run's threshold, averaged over seeds."""
    first = next(iter(runs[order[0]].values()))
    rust = first['counts'][:, [0, 2]].sum(1) / first['counts'].sum(1)
    lines = [
        '| Photograph | Camera | Rust | ' + ' | '.join(order) + ' |',
        '|---|---|---|' + '---|' * len(order),
    ]
    per_model = {m: np.mean([iou(runs[m][s]['counts']) for s in sorted(runs[m])], 0) for m in order}
    for i, (pid, cam) in enumerate(zip(first['photos'], first['cameras'], strict=True)):
        cells = ' | '.join(f'{per_model[m][i]:.4f}' for m in order)
        lines.append(f'| {pid} | {cam} | {100 * rust[i]:.2f}% | {cells} |')
    return lines


def _camera_table(runs: dict, order: list[str]) -> list[str]:
    """Pooled test IoU of each camera's photographs, averaged over seeds."""
    first = next(iter(runs[order[0]].values()))
    cams = sorted(set(first['cameras'].tolist()))
    lines = [
        '| Camera | Photographs | ' + ' | '.join(order) + ' |',
        '|---|---|' + '---|' * len(order),
    ]
    for cam in cams:
        sel = first['cameras'] == cam
        cells = ' | '.join(
            f'{np.mean([iou(runs[m][s]["counts"][sel].sum(0)) for s in runs[m]]):.4f}'
            for m in order
        )
        lines.append(f'| {cam} | {int(sel.sum())} | {cells} |')
    return lines


def speed_table(scale_dir: str, order: list[str], window: int, scale: float) -> list[str]:
    """Time per photograph from NWRD_final's T4 benchmark of the same architectures (D5)."""
    bench = _read_json(
        os.path.join(
            os.path.dirname(scale_dir),
            'nwrd_final',
            f'benchmark_{os.path.basename(scale_dir)}.json',
        )
    )
    if not bench:
        return []
    rows = {r['model']: r for r in bench['results']}
    per_photo = {
        k: len(grid(*scaled_shape(h, w, scale), window)) for k, (h, w) in PHOTO_SIZES.items()
    }
    lines = [
        '',
        f'## Speed ({bench["env"]["device"]}, measured by NWRD_final; results/nwrd_final/)',
        '',
        'Speed depends on the architecture and the window, not on the photographs, so it is not '
        'measured again. Seconds per photograph = windows per photograph / fp16 throughput at '
        'batch 8.',
        '',
        '| Model | Params (M) | GFLOPs per window | Windows/s | '
        + ' | '.join(f's per photograph: {k} ({n} windows)' for k, n in per_photo.items())
        + ' |',
        '|---|---|---|---|' + '---|' * len(per_photo),
    ]
    for m in (m for m in order if m in rows):
        r = rows[m]
        fps = r['gpu_throughput_img_s_bs8_fp16']
        lines.append(
            f'| {m} | {r["params_M"]:.1f} | {r["GFLOPs"]:.1f} | {fps:.0f} | '
            + ' | '.join(f'{n / fps:.1f}' for n in per_photo.values())
            + ' |'
        )
    return lines


def write_comparison(scale_dir: str, runs: dict, summary: dict, tests: dict) -> str:
    """Writes <scale_dir>/comparison.md and returns its path."""
    order = _order(summary)
    any_m = next(iter(next(iter(runs.values())).values()))['metrics']
    pp = any_m['paper_patches']
    L = [
        f"# CANet-B4 against the NWRD paper's baselines on the public NWRD release: "
        f'{os.path.basename(os.path.abspath(scale_dir))}',
        '',
        f'{any_m["window"]}px windows of photographs at {any_m["scale"]:g}× scale. Test: the '
        f"release's {any_m['test_photos']} test photographs, every pixel scored once at native "
        f'resolution ({100 * any_m["test_rust_pixel_fraction"]:.2f}% rust). Threshold chosen per '
        f'run on the {any_m["val_photos"]} validation photographs (IoU-maximising, grid '
        '0.01–0.99), then frozen for test. Metrics pooled over all test pixels; mean ± sample sd '
        'over seeds.',
        '',
        '## Accuracy (test photographs, validation-selected threshold)',
        '',
        '| Model | Seeds | ' + ' | '.join(METRICS) + ' | IoU @0.5 | Threshold |',
        '|---|---|' + '---|' * (len(METRICS) + 2),
    ]
    for m in order:
        r = summary[m]
        cells = [_fmt(*r['test@thr'][k], r['n']) for k in METRICS]
        thr = f'{r["threshold"][0]:.2f}' + (' ⚠ grid edge' if r['threshold_at_grid_edge'] else '')
        L.append(
            f'| {_name(m)} | {r["n"]} | '
            + ' | '.join(cells)
            + f' | {_fmt(*r["test@0.5"]["IoU"], r["n"])} | {thr} |'
        )
    L += [
        '',
        '## Each run',
        '',
        '| Model | Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold '
        '| Best epoch | Train min/epoch |',
        '|---|---|' + '---|' * 9,
    ]
    for m in order:
        for s in sorted(runs[m]):
            met, h = runs[m][s]['metrics'], runs[m][s]['history']['history']
            t = met['test@thr']
            L.append(
                f'| {m} | {s} | '
                + ' | '.join(f'{t[k]:.4f}' for k in ('IoU', 'Dice', 'Precision', 'Recall'))
                + f' | {met["test@0.5"]["IoU"]:.4f} | {met["val@thr"]["IoU"]:.4f} | '
                f'{met["threshold"]:.2f} | {met["best_epoch"]} | '
                f'{np.mean(h["epoch_seconds"]) / 60:.1f} |'
            )
    L += ['', "## Each test photograph (IoU at each run's threshold, mean over seeds)", '']
    L += _photo_table(runs, order)
    L += ['', '## By camera (pooled test IoU, mean over seeds)', '']
    L += _camera_table(runs, order)
    if tests:

        def ci(b):
            return f'[{b["ci95"][0]:+.4f}, {b["ci95"][1]:+.4f}] (p {b["p_two_sided"]:.4f})'

        L += [
            '',
            '## Is the difference real? (CANet-B4 minus baseline, test photographs)',
            '',
            '| Baseline | Paired seeds | ΔIoU | 95% CI, resampling photographs | ΔDice | '
            'ΔDice 95% CI | Seed-level t-test p | CANet-B4 higher (seeds) | Verdict |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for base, t in tests.items():
            b, tt, n = t['bootstrap_photo'], t['seed_ttest'], len(t['seeds'])
            reading = f'CANet-B4 {VERDICTS[t["verdict"]]} {base}'
            if n < 3:
                reading += f' (interim: {n} of 3 seeds)'
            p_seeds = f'{tt["p"]:.4f}' if tt else 'n/a (1 seed)'
            L.append(
                f'| {base} | {n} | {b["IoU"]["delta"]:+.4f} | {ci(b["IoU"])} | '
                f'{b["Dice"]["delta"]:+.4f} | {ci(b["Dice"])} | {p_seeds} | '
                f'{t["canet_wins"]} of {n} | {reading} |'
            )
        L += [
            '',
            'Verdict: the decision rule fixed before any result existed (docs/protocol.md §5). '
            'CANet-B4 *improves on* a baseline if its mean test IoU is higher, the 95% CI of '
            'ΔIoU that resamples test photographs excludes 0, and it is higher on at least 2 of 3 '
            'seeds; it *is worse* if that CI lies below 0; otherwise the two are *comparable*. '
            'The interval captures test-set sampling, not training-run variance; the seed-level '
            't-test does, with little power at 3 seeds.',
        ]
    if pp['patches'] == PAPER_TEST_PATCHES:
        patches = (
            f'The {pp["patches"]:,} 128px patches of the half-size test photographs that the '
            f"paper's test.py scores ({pp['rust_patches']} with rust), in its order."
        )
    else:
        patches = (
            f"A partial test set: {pp['patches']:,} of the paper's {PAPER_TEST_PATCHES:,} test "
            f'patches ({pp["rust_patches"]} with rust), so the published rows are not comparable '
            'with ours.'
        )
    halved = 'resized by area to half size' if any_m['scale'] == 1 else 'taken at half size'
    L += [
        '',
        "## On the NWRD paper's own test patches",
        '',
        f'{patches} Our predictions are those scored above, {halved}; masks halved by area.',
        '',
        '| Result | Precision | Recall | F1 / Dice | IoU |',
        '|---|---|---|---|---|',
    ]
    for p in PAPER:
        s = p['scores']
        L.append(
            f'| Paper, {p["table"]}, UNet + APF, {p["patches"]} | '
            + ' | '.join(
                f'{s[k]:.3f}' if k in s else '–' for k in ('Precision', 'Recall', 'F1', 'IoU')
            )
            + ' |'
        )
    for m in order:
        for thr, kind, label in PAPER_ROWS:
            s = summary[m]['paper'][thr][kind]
            f1 = s['F1'] if 'F1' in s else s['Dice']  # the authors' averaging has no IoU
            iou_cell = f'{s["IoU"][0]:.3f}' if 'IoU' in s else '–'
            L.append(
                f'| {m}: {label} | {s["Precision"][0]:.3f} | {s["Recall"][0]:.3f} | '
                f'{f1[0]:.3f} | {iou_cell} |'
            )
    L += [
        '',
        "The paper's rows are as published, on the same photographs. Table 3 is its UNet trained "
        'with APF (focal loss, 500 epochs, 128px patches at stride 32) and scored by its test.py '
        'on all the patches. Table 4 is a separately trained UNet (stride 128, background-only '
        'patches removed from training as well as test, paper §5), and its IoU was not computed '
        "by test.py. The paper's training set held twins of test photographs 80 and 89 "
        '(docs/dataset.md); no model here trained on a twin of a validation or test photograph. '
        "The `unet` rows are the paper's architecture re-trained under this protocol, so "
        'CANet-B4 against `unet` is the like-for-like comparison.',
    ]
    L += speed_table(scale_dir, order, any_m['window'], any_m['scale'])
    L += ['', '## Models', ''] + [f'- `{m}` — {MODELS.get(m, m)}' for m in order]
    path = os.path.join(scale_dir, 'comparison.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return path


def write_model_summary(out_dir: str, model: str) -> str | None:
    """Writes <out_dir>/<model>/summary.md: one model's runs, seed by seed, and its mean ± sd.
    Returns its path, or None while the model has no finished run."""
    runs = load_runs(out_dir).get(model)
    if not runs:
        return None
    r = summarise({model: runs})[model]
    keys = ('IoU', 'Dice', 'Precision', 'Recall')
    any_m = next(iter(runs.values()))['metrics']
    L = [
        f'# {model}: {MODELS.get(model, model)}',
        '',
        f'{any_m["window"]}px windows at {any_m["scale"]:g}× scale. Test: '
        f'{any_m["test_photos"]} photographs, every pixel scored once at native resolution. '
        'Threshold chosen per run on validation; metrics pooled over all test pixels. The paper '
        "patches column is F1 on the NWRD paper's test patches with the authors' averaging at "
        '0.5.',
        '',
        '| Seed | Test IoU | Dice | Precision | Recall | IoU @0.5 | Val IoU | Threshold | '
        'Best epoch | Paper patches F1 | Train min/epoch |',
        '|---|' + '---|' * 10,
    ]
    for s in sorted(runs):
        m, h = runs[s]['metrics'], runs[s]['history']['history']
        t = m['test@thr']
        L.append(
            f'| {s} | '
            + ' | '.join(f'{t[k]:.4f}' for k in keys)
            + f' | {m["test@0.5"]["IoU"]:.4f} | {m["val@thr"]["IoU"]:.4f} | '
            f'{m["threshold"]:.2f} | {m["best_epoch"]} | '
            f'{m["paper_patches"]["@0.5"]["authors"]["F1"]:.4f} | '
            f'{np.mean(h["epoch_seconds"]) / 60:.1f} |'
        )
    means = [r['test@thr'][k] for k in keys] + [r['test@0.5']['IoU'], r['val@thr']['IoU']]
    L.append(
        '| **mean ± sd** | '
        + ' | '.join(_fmt(*v, r['n']) for v in means)
        + f' | {r["threshold"][0]:.2f} | – | '
        f'{_fmt(*r["paper"]["@0.5"]["authors"]["F1"], r["n"])} | '
        f'{r["train_min_per_epoch"][0]:.1f} |'
    )
    L += [
        '',
        "Side by side with the other models: comparison.md of the scale's results folder "
        '(python -m nwrd_public.compare results).',
    ]
    path = os.path.join(out_dir, model, 'summary.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    return path


def plot_training_curves(runs: dict, path: str) -> None:
    """Validation loss and IoU per epoch: mean over seeds, band from min to max."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for model in _order(runs):
        seeds = runs[model]
        for key, ax in (('val_loss', axes[0]), ('val_iou', axes[1])):
            curves = [seeds[s]['history']['history'][key] for s in sorted(seeds)]
            n = min(map(len, curves))
            arr = np.array([c[:n] for c in curves])
            x = np.arange(1, n + 1)
            (line,) = ax.plot(x, arr.mean(0), label=model, lw=2 if model == OURS else 1.4)
            if len(curves) > 1:
                ax.fill_between(x, arr.min(0), arr.max(0), color=line.get_color(), alpha=0.15)
    axes[0].set_title('Validation loss (windows)')
    axes[1].set_title('Validation IoU (whole photographs, threshold 0.5)')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare(scale_dir: str, log=print) -> dict | None:
    """Writes comparison.md, comparison.json, the training curves and each model's summary."""
    runs = load_runs(scale_dir)
    if not runs:
        log(f'[compare] no finished runs in {scale_dir}')
        return None
    incomplete = [
        r['dir']
        for seeds in runs.values()
        for r in seeds.values()
        if not {'history', 'counts', 'paper'} <= set(r)
    ]
    if incomplete:
        raise FileNotFoundError(
            f'history.json, test_counts.npz or paper_patches.npz missing in {incomplete}'
        )
    summary = summarise(runs)
    tests = significance(runs)
    plot_training_curves(runs, os.path.join(scale_dir, 'figures', 'training_curves.png'))
    md_path = write_comparison(scale_dir, runs, summary, tests)
    for model in runs:
        write_model_summary(scale_dir, model)
    with open(os.path.join(scale_dir, 'comparison.json'), 'w') as f:
        json.dump({'summary': summary, 'significance': tests}, f, indent=2)
    with open(md_path, encoding='utf-8') as f:
        log(f.read())
    return summary


SUMMARY_COLUMNS = (
    'scale', 'model', 'seed', 'best_epoch', 'threshold', 'threshold_at_grid_edge',
    'test_iou', 'test_dice', 'test_precision', 'test_recall', 'test_specificity',
    'test_accuracy', 'test_iou_at_0.5', 'val_iou',
    'paper_authors_precision_at_0.5', 'paper_authors_recall_at_0.5', 'paper_authors_f1_at_0.5',
    'paper_authors_rust_only_f1_at_0.5', 'paper_pooled_iou', 'paper_rust_only_iou',
    'train_min_per_epoch',
)  # fmt: skip


def summary_rows(results_root: str) -> list[dict]:
    """One row per (scale, model, seed) found under `results_root`."""
    rows = []
    for scale in SCALES:
        runs = load_runs(os.path.join(results_root, scale))
        for model in _order(runs):
            for seed in sorted(runs[model]):
                m, h = runs[model][seed]['metrics'], runs[model][seed].get('history')
                t, p = m['test@thr'], m['paper_patches']
                rows.append(
                    {
                        'scale': scale,
                        'model': model,
                        'seed': seed,
                        'best_epoch': m['best_epoch'],
                        'threshold': m['threshold'],
                        'threshold_at_grid_edge': m['threshold_at_grid_edge'],
                        **{f'test_{k.lower()}': t[k] for k in METRICS},
                        'test_iou_at_0.5': m['test@0.5']['IoU'],
                        'val_iou': m['val@thr']['IoU'],
                        'paper_authors_precision_at_0.5': p['@0.5']['authors']['Precision'],
                        'paper_authors_recall_at_0.5': p['@0.5']['authors']['Recall'],
                        'paper_authors_f1_at_0.5': p['@0.5']['authors']['F1'],
                        'paper_authors_rust_only_f1_at_0.5': p['@0.5']['authors_rust_only']['F1'],
                        'paper_pooled_iou': p['@thr']['pooled']['IoU'],
                        'paper_rust_only_iou': p['@thr']['rust_only']['IoU'],
                        'train_min_per_epoch': (
                            float(np.mean(h['history']['epoch_seconds']) / 60) if h else ''
                        ),
                    }
                )
    return rows


def write_summary_csv(results_root: str) -> str:
    """results/summary.csv: one row per (scale, model, seed), for a spreadsheet."""
    path = os.path.join(results_root, 'summary.csv')
    rows = [
        {k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()}
        for row in summary_rows(results_root)
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('results', help='the results folder holding 512px/ and 128px/')
    a = ap.parse_args(argv)
    for scale in SCALES:
        if os.path.isdir(os.path.join(a.results, scale)):
            compare(os.path.join(a.results, scale))
    print('wrote', write_summary_csv(a.results))


if __name__ == '__main__':
    # The reports contain ± and ≈, which a redirected stdout on Windows (cp1252) cannot encode.
    sys.stdout.reconfigure(encoding='utf-8')
    main()
