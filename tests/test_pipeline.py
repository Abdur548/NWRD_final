"""End-to-end runs of the pipeline on CPU with the miniature release (slow: minutes)."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from nwrd_public.dataset import read_photo
from nwrd_public.run import main

from .conftest import ROOT

pytestmark = pytest.mark.slow

FAST = ['--window', '128', '--batch-size', '2', '--num-workers', '0', '--pretrained', 'false',
        '--limit', '2']  # fmt: skip


def _read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def _data(release, tmp_path):
    return ['--data-root', release.root, '--manifest', release.manifest,
            '--window-dir', str(tmp_path / 'windows')]  # fmt: skip


def _history(run_dir):
    h = json.loads(_read(os.path.join(run_dir, 'history.json')))
    h['history'].pop('epoch_seconds')  # wall clock
    return h


def _weights(run_dir):
    path = os.path.join(run_dir, 'best.pt')
    return torch.load(path, map_location='cpu', weights_only=True)['model']


def _same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_full_run(synthetic_release, tmp_path):
    out = tmp_path / 'runs'
    args = [*_data(synthetic_release, tmp_path), '--out', str(out), *FAST, '--model', 'canet_b4',
            '--seeds', '1', '2', '--epochs', '2', '--freeze-encoder-epochs', '1',
            '--bg-keep-ratio', '0.5']  # fmt: skip
    assert main(args) == 0
    log = _read(out / 'log.txt')
    started = [
        line.split('] data:')[0].split('[')[1] for line in log.splitlines() if '] data:' in line
    ]
    assert started == ['canet_b4 s1', 'canet_b4 s2']
    assert 'all 200 files match the manifest' in log

    run = out / 'canet_b4' / 'seed1'
    m = json.loads(_read(run / 'metrics.json'))
    assert 0.01 <= m['threshold'] <= 0.99 and m['test_photos'] == 2
    assert (m['window'], m['scale'], m['model']) == (128, 1.0, 'canet_b4')
    assert set(m['paper_patches']['@thr']) == {
        'pooled',
        'rust_only',
        'authors',
        'authors_rust_only',
    }
    assert not (run / 'last.pt').exists() and (run / 'best.pt').exists()

    # Every native pixel of each test photograph is scored once.
    counts = np.load(run / 'test_counts.npz')
    assert counts['photos'].tolist() == ['25', '29']
    for pid, c in zip(counts['photos'], counts['at_thr'], strict=True):
        assert c.sum() == read_photo(synthetic_release.root, 'test', pid)[1].size
    # The paper's patches: photograph by photograph in the authors' order, row by row.
    paper = np.load(run / 'paper_patches.npz')
    keys = list(zip(paper['photo'].tolist(), paper['row'].tolist(), paper['col'].tolist(),
                    strict=True))  # fmt: skip
    assert keys == sorted(keys) and len(keys) == m['paper_patches']['patches'] > 0
    assert (paper['at_05'].sum(1) == 128 * 128).all()

    windows = json.loads(_read(out / 'windows.json'))
    assert windows['params']['photos'] == ['3', '5'] and windows['sample'] == len(windows['files'])
    assert json.loads(_read(out / 'config.json'))['preset'] == '512px'
    assert '| **mean ± sd** |' in _read(out / 'canet_b4' / 'summary.md')

    # Re-running is a no-op, and the cut is reused.
    assert main(args) == 0
    assert 'windows: reusing' in _read(out / 'log.txt')
    # Writing the summary alone never rewrites what the run recorded; from the command line it
    # must also survive a stdout that cannot encode the report's ±, as a redirected one on Windows.
    config, log = _read(out / 'config.json'), _read(out / 'log.txt')
    cli = subprocess.run(
        [sys.executable, '-m', 'nwrd_public.run', '--stage', 'summary', '--out', str(out)],
        cwd=ROOT,
        capture_output=True,
        env={**os.environ, 'PYTHONIOENCODING': 'cp1252'},
    )
    assert cli.returncode == 0, cli.stderr.decode(errors='replace')
    assert _read(out / 'config.json') == config and _read(out / 'log.txt') == log


def test_full_run_at_half_scale(synthetic_release, tmp_path):
    out = tmp_path / 'runs'
    args = [*_data(synthetic_release, tmp_path), '--out', str(out), '--preset', '128px', *FAST,
            '--model', 'deeplabv3plus_r50', '--seeds', '1', '--epochs', '1']  # fmt: skip
    assert main(args) == 0
    m = json.loads(_read(out / 'deeplabv3plus_r50' / 'seed1' / 'metrics.json'))
    assert (m['window'], m['scale']) == (128, 0.5)
    counts = np.load(out / 'deeplabv3plus_r50' / 'seed1' / 'test_counts.npz')
    for pid, c in zip(counts['photos'], counts['at_05'], strict=True):
        assert c.sum() == read_photo(synthetic_release.root, 'test', pid)[1].size  # native pixels
    config = json.loads(_read(out / 'config.json'))
    assert config['preset'] == '128px' and config['scale'] == 0.5


def test_training_is_reproducible_and_resumes_exactly(synthetic_release, tmp_path):
    """Same seed, same outputs; a run stopped by the time budget and resumed sees the same
    batches and augmentations as one that ran straight through."""
    base = [*_data(synthetic_release, tmp_path), *FAST, '--model', 'canet_b4', '--seeds', '1',
            '--epochs', '3', '--freeze-encoder-epochs', '1', '--stage', 'train']  # fmt: skip
    straight, again, resumed = (tmp_path / n for n in ('straight', 'again', 'resumed'))
    assert main(base + ['--out', str(straight)]) == 0
    assert main(base + ['--out', str(again)]) == 0
    assert main(base + ['--out', str(resumed), '--time-budget-hours', '1e-9']) == 3  # 1 epoch
    run = resumed / 'canet_b4' / 'seed1'
    assert (run / 'last.pt').exists() and not (run / 'TRAINED').exists()
    assert len(_history(run)['history']['train_loss']) == 1
    assert main(base + ['--out', str(resumed)]) == 0

    ref = straight / 'canet_b4' / 'seed1'
    for other in (again, resumed):
        assert _history(other / 'canet_b4' / 'seed1') == _history(ref)
        assert _same_weights(_weights(other / 'canet_b4' / 'seed1'), _weights(ref))


def test_augmentation_is_seeded_in_worker_processes(synthetic_release, tmp_path):
    base = [*_data(synthetic_release, tmp_path), *FAST, '--model', 'unet', '--seeds', '1',
            '--epochs', '1', '--stage', 'train', '--num-workers', '2']  # fmt: skip
    a, b = tmp_path / 'a', tmp_path / 'b'
    assert main(base + ['--out', str(a)]) == 0
    assert main(base + ['--out', str(b)]) == 0
    assert _history(a / 'unet' / 'seed1') == _history(b / 'unet' / 'seed1')


def test_resume_survives_checkpoints_loaded_off_the_cpu(synthetic_release, tmp_path, monkeypatch):
    # On a GPU, loading last.pt onto the compute device put the saved RNG state on the GPU,
    # and torch.set_rng_state rejects that, so every resume crashed. Emulate it: whatever is
    # not loaded explicitly to 'cpu' lands on another device ('meta' stands in for cuda).
    from nwrd_public import train

    out = tmp_path / 'runs'
    base = [*_data(synthetic_release, tmp_path), '--out', str(out), *FAST, '--model', 'canet_b4',
            '--seeds', '1', '--epochs', '2', '--stage', 'train']  # fmt: skip
    assert main(base + ['--time-budget-hours', '1e-9']) == 3
    real_load = torch.load

    def load_like_on_gpu(path, map_location=None, **kw):
        ck = real_load(path, map_location='cpu', **kw)
        if map_location != 'cpu' and isinstance(ck, dict) and 'rng_cpu' in ck:
            ck['rng_cpu'] = ck['rng_cpu'].to('meta')
        return ck

    monkeypatch.setattr(train.torch, 'load', load_like_on_gpu)
    assert main(base) == 0
    assert (out / 'canet_b4' / 'seed1' / 'TRAINED').exists()


def test_time_budget_counts_from_session_start(synthetic_release, tmp_path):
    # Each run fits the budget on its own clock, but the session does not: the second seed must
    # not start. Measuring per run let a Kaggle session overrun its 12 h limit.
    out = tmp_path / 'runs'
    args = [*_data(synthetic_release, tmp_path), '--out', str(out), *FAST, '--model', 'unet',
            '--seeds', '1', '2', '--epochs', '1', '--stage', 'train',
            '--time-budget-hours', '1e-9']  # fmt: skip
    assert main(args) == 3
    assert (out / 'unet' / 'seed1' / 'TRAINED').exists()
    assert not (out / 'unet' / 'seed2').exists() or not any((out / 'unet' / 'seed2').iterdir())
    assert main(args) == 0  # the next session's first run always makes progress
    assert (out / 'unet' / 'seed2' / 'TRAINED').exists()


def test_a_changed_file_stops_before_training(synthetic_release, tmp_path):
    copy = tmp_path / 'release'
    shutil.copytree(synthetic_release.root, copy)
    photo = copy / 'train' / 'images' / '3.jpg'
    data = bytearray(photo.read_bytes())
    data[-3] ^= 0xFF
    photo.write_bytes(bytes(data))
    out = tmp_path / 'runs'
    args = ['--data-root', str(copy), '--manifest', synthetic_release.manifest, '--out', str(out),
            '--window-dir', str(tmp_path / 'windows'), *FAST, '--seeds', '1',
            '--epochs', '1']  # fmt: skip
    with pytest.raises(RuntimeError, match='1 changed'):
        main(args)
    assert not (out / 'canet_b4').exists() and not (tmp_path / 'windows').exists()
    assert 'CRASHED' in _read(out / 'log.txt')


def test_models_side_by_side(synthetic_release, tmp_path):
    """Two models' Kaggle outputs collected into one results folder and compared."""
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import collect_results

    results = tmp_path / 'results'
    for model in ('canet_b4', 'unet'):
        kernel = tmp_path / 'kaggle' / model / 'runs'
        args = [*_data(synthetic_release, tmp_path), '--out', str(kernel), *FAST, '--model', model,
                '--seeds', '1', '2', '--epochs', '1', '--freeze-encoder-epochs', '0']  # fmt: skip
        assert main(args) == 0
        collect_results.collect(str(kernel.parent), str(results), preset='512px')
    assert sorted(os.listdir(results / '512px')) == [
        'canet_b4', 'comparison.json', 'comparison.md', 'figures', 'unet'
    ]  # fmt: skip
    assert not list(results.rglob('*.pt')) and not list(results.rglob('TRAINED'))
    md = _read(results / '512px' / 'comparison.md')
    assert '| **canet_b4** (ours) |' in md and '| Verdict |' in md
    assert 'CANet-B4 is' in md and '(interim: 2 of 3 seeds)' in md
    assert "On the NWRD paper's own test patches" in md and '| 25 |' in md
    comp = json.loads(_read(results / '512px' / 'comparison.json'))
    assert comp['significance']['unet']['seeds'] == [1, 2]
    rows = _read(results / 'summary.csv').splitlines()
    assert len(rows) == 1 + 4 and rows[1].startswith('512px,canet_b4,1,')
