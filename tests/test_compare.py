"""The photograph bootstrap, the pre-registered decision rule, the side-by-side report and the
speed table."""

import json
import os

import numpy as np
import pytest

from nwrd_public.compare import (
    compare,
    paired_bootstrap,
    speed_table,
    verdict,
    write_summary_csv,
)
from nwrd_public.evaluate import paper_scores
from nwrd_public.metrics import scores

from .conftest import ROOT

PHOTOS = ['105', '123', '25', '29']


def _runs(counts, photos):
    return {1: {'photos': np.array(photos), 'counts': np.array(counts, np.int64)}}


def _fake_run(run_dir, model: str, seed: int, patches: int) -> None:
    """The files a finished run leaves, with random but consistent counts."""
    rng = np.random.default_rng(seed + len(model))
    counts = np.stack([rng.integers(10, 1000, 4) for _ in PHOTOS])
    paper = rng.integers(0, 500, (patches, 4))
    rust = paper[:, 0] + paper[:, 2] > 0
    test = scores(*counts.sum(0))
    run_dir.mkdir(parents=True)
    metrics = {
        'model': model, 'window': 512, 'scale': 1.0, 'best_epoch': 3, 'threshold': 0.5,
        'threshold_at_grid_edge': False, 'val@thr': test, 'val@0.5': test, 'test@thr': test,
        'test@0.5': test, 'val_photos': 9, 'test_photos': len(PHOTOS),
        'test_rust_pixel_fraction': 0.006,
        'paper_patches': {'patches': patches, 'rust_patches': int(rust.sum()),
                          '@thr': paper_scores(paper, rust), '@0.5': paper_scores(paper, rust)},
    }  # fmt: skip
    (run_dir / 'metrics.json').write_text(json.dumps(metrics))
    history = {'val_loss': [0.5, 0.4], 'val_iou': [0.2, 0.3], 'epoch_seconds': [60.0, 61.0]}
    (run_dir / 'history.json').write_text(json.dumps({'history': history}))
    cameras = np.array(['NIKON D3200', 'PAR-LX1M', 'NIKON D3200', 'PAR-LX1M'])
    np.savez(run_dir / 'test_counts.npz', photos=np.array(PHOTOS), cameras=cameras,
             at_thr=counts, at_05=counts)  # fmt: skip
    np.savez(run_dir / 'paper_patches.npz', at_thr=paper, at_05=paper, rust=rust)


@pytest.mark.parametrize('patches', [2737, 690])
def test_side_by_side_report(tmp_path, patches):
    scale_dir = tmp_path / '512px'
    for model in ('canet_b4', 'unet', 'deeplabv3plus_r50'):
        for seed in (42, 43):
            _fake_run(scale_dir / model / f'seed{seed}', model, seed, patches)
    compare(str(scale_dir), log=lambda _msg: None)
    md = (scale_dir / 'comparison.md').read_text(encoding='utf-8')
    assert md.index('| **canet_b4** (ours) |') < md.index('| deeplabv3plus_r50 |')
    for pid in PHOTOS:  # one row per test photograph
        assert f'| {pid} |' in md
    assert '| NIKON D3200 | 2 |' in md and '| PAR-LX1M | 2 |' in md
    for base in ('unet', 'deeplabv3plus_r50'):
        assert f' {base} (interim: 2 of 3 seeds) |' in md
    assert 'Table 4 is a separately trained UNet' in md
    if patches == 2737:
        assert "The 2,737 128px patches of the half-size test photographs that the paper's" in md
        assert 'partial' not in md
    else:
        assert "A partial test set: 690 of the paper's 2,737 test patches" in md
    summary = json.loads((scale_dir / 'comparison.json').read_text())
    assert summary['significance']['unet']['seeds'] == [42, 43]
    assert (scale_dir / 'unet' / 'summary.md').exists()


def test_photograph_bootstrap():
    photos = [str(p) for p in range(10)]
    base = [(70, 15, 15, 900)] * 10
    better = _runs([(80, 10, 10, 900)] * 10, photos)
    b = paired_bootstrap(better, _runs(base, photos), [1])['IoU']
    assert b['units'] == 10 and b['delta'] == pytest.approx(80 / 100 - 70 / 100)
    assert b['ci95'][0] > 0  # better on every photograph: the interval excludes 0
    mixed = _runs([(90, 5, 5, 900)] * 5 + [(50, 25, 25, 900)] * 5, photos)
    m = paired_bootstrap(mixed, _runs(base, photos), [1])['IoU']
    assert m['ci95'][0] < 0 < m['ci95'][1]


def test_bootstrap_refuses_unpaired_test_sets():
    counts = np.ones((2, 4), np.int64)
    with pytest.raises(ValueError, match='different photographs'):
        paired_bootstrap(_runs(counts, ['1', '2']), _runs(counts, ['2', '1']), [1])


@pytest.mark.parametrize(
    'delta, ci, wins, expected',
    [
        (0.02, (0.005, 0.04), 3, 'improves'),
        (0.02, (0.005, 0.04), 1, 'comparable'),  # CI excludes 0 but too few seed wins
        (0.01, (-0.01, 0.03), 3, 'comparable'),  # CI includes 0
        (-0.04, (-0.08, -0.006), 0, 'worse'),
        (-0.02, (-0.042, 0.019), 0, 'comparable'),
    ],
)
def test_decision_rule(delta, ci, wins, expected):
    assert verdict({'delta': delta, 'ci95': list(ci)}, wins) == expected


@pytest.mark.parametrize(
    'scale, window, nikon, huawei', [('512px', 512, 96, 63), ('128px', 128, 384, 252)]
)
def test_seconds_per_photograph_from_the_t4_benchmark(scale, window, nikon, huawei):
    lines = speed_table(os.path.join(ROOT, 'results', scale), ['canet_b4', 'unet'], window,
                        1.0 if scale == '512px' else 0.5)  # fmt: skip
    text = '\n'.join(lines)
    assert f'({nikon} windows)' in text and f'({huawei} windows)' in text
    canet = next(line for line in lines if line.startswith('| canet_b4 |'))
    fps = float(canet.split('|')[4])
    assert float(canet.split('|')[5]) == pytest.approx(nikon / fps, abs=0.1)


def test_summary_csv_of_an_empty_results_folder(tmp_path):
    path = write_summary_csv(str(tmp_path))
    with open(path, encoding='utf-8') as f:
        assert f.read().startswith('scale,model,seed,')
