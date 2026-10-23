"""Tabulates the 128px experiment against the NWRD paper and the 512px experiment.

    python scripts/compare_scales.py results/128px/final results/512px/final
    python scripts/compare_scales.py results/128px/final results/512px/final \
        --out results/analysis/scales.md

Writes markdown: CANet-B4 vs the paper's UNet and DeepLabV3+ R50 at the paper's scale
(photograph-level bootstrap and the pre-registered verdict, as recorded in comparison.json), the
paper's published numbers beside
the matching test sets, how each model changes between scales, and efficiency per input and per
photograph. Reads only files the pipeline writes.
"""

import argparse
import glob
import json
import math
import os

import numpy as np

from nwrd.compare import OURS, PAPER, VERDICTS, mean_std

BASES = ('unet', 'deeplabv3plus_r50')  # in the order they are reported
NAMES = {'unet': "the paper's UNet", 'deeplabv3plus_r50': 'DeepLabV3+ R50'}
KEYS = ('Precision', 'Recall', 'Dice', 'IoU')
# Every photograph of this study is 6016×4000 (docs/protocol.md §3.1).
PHOTO = (6016, 4000)


def _read_json(path: str):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load(run_dir: str) -> dict:
    comp = _read_json(os.path.join(run_dir, 'comparison.json'))
    seeds = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*', 'seed*', 'metrics.json'))):
        model = os.path.basename(os.path.dirname(os.path.dirname(path)))
        seed = int(os.path.basename(os.path.dirname(path))[len('seed') :])
        seeds.setdefault(model, {})[seed] = _read_json(path)
    tiling = os.path.join(run_dir, 'tiling.json')
    return {
        'comp': comp,
        'seeds': seeds,
        'bench': {r['model']: r for r in (comp.get('benchmark') or {}).get('results', [])},
        'tiling': _read_json(tiling) if os.path.exists(tiling) else None,
    }


def fmt(xs) -> str:
    m, s = mean_std(xs)
    return f'{m:.3f} ± {s:.3f}' if len(xs) > 1 else f'{m:.3f}'


def per_photo(size: int, scale: float) -> int:
    """Number of size-px inputs one photograph, resized by `scale`, is cut into once padded to
    whole inputs, as the dataset and the app cut it."""
    return math.ceil(PHOTO[0] * scale / size) * math.ceil(PHOTO[1] * scale / size)


def report(r128: dict, r512: dict) -> str:
    models = [m for m in (OURS, *BASES) if m in r128['seeds']]
    L = []
    t = r128['tiling']
    if t:
        test, train = t['splits']['test'], t['splits']['train']
        L += [
            f'Tiles: {t["tile"]}px at {t["scale"]:g}× scale. Test: {test["tiles"]} tiles '
            f'({test["rust_tiles"]} with rust) from {test["patches"]} patches. Rust pixel share, '
            f'train: {100 * train["rust_pixel_fraction_patches"]:.2f}% of patches, '
            f'{100 * train["rust_pixel_fraction_tiles"]:.2f}% of tiles.',
            '',
        ]

    L += [
        '## Accuracy at 128px (test tiles, threshold chosen on validation)',
        '',
        '| Model | Seeds | IoU | Dice / F1 | Precision | Recall | IoU per seed |',
        '|---|---|---|---|---|---|---|',
    ]
    for m in models:
        runs = r128['seeds'][m]
        cols = [
            fmt([runs[s]['test@thr'][k] for s in runs])
            for k in ('IoU', 'Dice', 'Precision', 'Recall')
        ]
        per_seed = ', '.join(f'{s}: {runs[s]["test@thr"]["IoU"]:.3f}' for s in sorted(runs))
        L.append(f'| {m} | {len(runs)} | ' + ' | '.join(cols) + f' | {per_seed} |')
    for base in BASES:
        sig = r128['comp']['significance'].get(base)
        if not sig:
            continue
        b, tt, n = sig['bootstrap_photo']['IoU'], sig['seed_ttest'], len(sig['seeds'])
        L += [
            '',
            f'CANet − {NAMES[base]}: ΔIoU {b["delta"]:+.4f}, 95% CI [{b["ci95"][0]:+.4f}, '
            f'{b["ci95"][1]:+.4f}] resampling test photographs (p {b["p_two_sided"]:.3f}); '
            f'CANet higher on {sig["canet_wins"]} of {n} seeds'
            + (f'; seed-level t-test p {tt["p"]:.3f}.' if tt else '.'),
            f'Pre-registered decision rule at 128px: CANet-B4 **{VERDICTS[sig["verdict"]]}** '
            f'{NAMES[base]}' + ('' if n >= 3 else f' (interim: {n} of 3 seeds)') + '.',
        ]

    L += [
        '',
        "## Against the paper's published UNet + APF",
        '',
        '| Source | Test set | Precision | Recall | F1 / Dice | IoU |',
        '|---|---|---|---|---|---|',
    ]
    for p in PAPER:
        L.append(
            f'| Paper, {p["table"]} | {p["test_set"]} | '
            + ' | '.join(f'{p["scores"][k]:.3f}' if k in p['scores'] else '–' for k in KEYS)
            + ' |'
        )
        ours = 'all test tiles' if p['split'] == 'test@thr' else 'rust tiles only'
        for m in models:
            runs = r128['seeds'][m]
            L.append(
                f'| {m}, 128px | {ours} | '
                + ' | '.join(f'{np.mean([runs[s][p["split"]][k] for s in runs]):.3f}' for k in KEYS)
                + ' |'
            )
    L += [
        '',
        'Same patch size and downsampling as the paper. Still different: the photographs (the '
        "paper's are the public NWRD release; this study's are not, docs/protocol.md §3.1), "
        "the paper's test split (random 80/10/10), its training (APF, RMSprop, 500 epochs, "
        'focal loss) and patch sampling, so these rows are context, not a paired comparison.',
    ]

    L += [
        '',
        '## Effect of scale on each model (test IoU, mean over seeds)',
        '',
        '| Model | 512px native | 128px at 0.5× | Change | Seeds (512 / 128) |',
        '|---|---|---|---|---|',
    ]
    for m in models:
        a512 = [x['test@thr']['IoU'] for x in r512['seeds'].get(m, {}).values()]
        a128 = [x['test@thr']['IoU'] for x in r128['seeds'][m].values()]
        if a512:
            L.append(
                f'| {m} | {fmt(a512)} | {fmt(a128)} | {np.mean(a128) - np.mean(a512):+.3f} | '
                f'{len(a512)} / {len(a128)} |'
            )

    L += ['', '## Efficiency', '']
    rows = []
    for label, run, size, scale in (('512px', r512, 512, 1.0), ('128px', r128, 128, 0.5)):
        summ = run['comp']['summary']
        for m in models:
            r = run['bench'].get(m)
            if not r:
                continue
            thr = {
                int(k.split('_bs')[1].split('_')[0]): v
                for k, v in r.items()
                if k.startswith('gpu_throughput_img_s_bs')
            }
            rows.append((label, m, size, scale, r, thr, summ.get(m, {}).get('train_min_per_epoch')))
    if rows:
        env = r128['comp']['benchmark']['env']
        L += [
            f'Measured on {env["device"]} (torch {env["torch"]}), random weights.',
            '',
            '| Scale | Model | Params (M) | GFLOPs | GPU fp32 bs1 (ms) | GPU fp16 bs1 (ms) | '
            'fp16 throughput (img/s) | CPU bs1 (ms) | Train min/epoch |',
            '|---|---|---|---|---|---|---|---|---|',
        ]
        for label, m, _size, _scale, r, thr, tr in rows:
            tp = ', '.join(f'{v:.0f} @bs{b}' for b, v in sorted(thr.items())) or '–'
            L.append(
                f'| {label} | {m} | {r["params_M"]:.1f} | {r["GFLOPs"]:.2f} | '
                f'{r.get("gpu_latency_ms_bs1_fp32", float("nan")):.1f} | '
                f'{r.get("gpu_latency_ms_bs1_fp16", float("nan")):.1f} | {tp} | '
                f'{r["cpu_latency_ms_bs1_fp32"]:.0f} | {f"{tr[0]:.1f}" if tr else "–"} |'
            )
        L += [
            '',
            f'Per {PHOTO[0]}×{PHOTO[1]} photograph ({per_photo(512, 1)} native 512px patches, or '
            f'{per_photo(128, 0.5)} tiles of 128px at 0.5×, padding included; GPU batched time '
            'uses the largest batch measured):',
            '',
            '| Scale | Model | TFLOPs / photo | GPU fp16 batched (ms / photo) | '
            'GPU fp32 bs1 (ms / photo) | CPU bs1 (s / photo) |',
            '|---|---|---|---|---|---|',
        ]
        for label, m, size, scale, r, thr, _tr in rows:
            n = per_photo(size, scale)
            batched = f'{1000 * n / thr[max(thr)]:.0f}' if thr else '–'
            fp32 = r.get('gpu_latency_ms_bs1_fp32')
            L.append(
                f'| {label} | {m} | {r["GFLOPs"] * n / 1000:.2f} | {batched} | '
                f'{f"{fp32 * n:.0f}" if fp32 else "–"} | '
                f'{r["cpu_latency_ms_bs1_fp32"] * n / 1000:.1f} |'
            )
    return '\n'.join(L) + '\n'


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('run128', help='results of the 128px experiment')
    ap.add_argument('run512', help='results of the main (512px) experiment')
    ap.add_argument('--out', help='write the markdown here instead of printing it')
    a = ap.parse_args()
    text = report(load(a.run128), load(a.run512))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write(text)
        print('wrote', a.out)
    else:
        print(text)


if __name__ == '__main__':
    main()
