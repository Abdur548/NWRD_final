"""The Kaggle runner must never push a resume that can silently restart from scratch, and never
mix up kernels, scales, versions or accounts."""

import itertools
import os
import re
import sys

import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import run_on_kaggle as k  # noqa: E402

KERNELS = [k.kernel(m, p) for m, p in itertools.product(k.MODELS, k.PRESETS)]


def test_one_kernel_per_model_and_scale():
    assert k.kernel('canet_b4', '512px')['slug'] == 'nwrd-public-canet-b4-512px'
    assert k.kernel('deeplabv3plus_r50', '128px')['slug'] == 'nwrd-public-deeplabv3plus-r50-128px'
    for key in ('slug', 'push_dir'):
        assert len({e[key] for e in KERNELS}) == 6, key
    assert len({k.output_dir(e, 'u', 1) for e in KERNELS}) == 6
    assert all(os.path.exists(e['notebook']) for e in KERNELS)


def test_kernel_metadata():
    e = k.kernel('unet', '128px')
    meta = k.kernel_metadata(e, 'u/' + e['slug'], resume=True)
    assert meta['title'] == 'NWRD public unet 128px' and meta['code_file'] == 'unet.ipynb'
    assert meta['dataset_sources'] == ['bcap52/nwrd-wheat'] and meta['kernel_sources'] == [
        'u/' + e['slug']
    ]
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
    assert k.kernel_metadata(e, 'u/x', resume=False)['kernel_sources'] == []


def test_pushed_copy_sets_scale_resume_and_budget():
    nb = k.kernel('unet', '512px')['notebook']
    fresh = k.notebook_source(nb)
    assert k.PRESET_LINE in fresh and k.EXPECT_OFF in fresh and k.BUDGET_LINE in fresh
    pushed = k.notebook_source(nb, '128px', resume=True, budget_hours=6)
    assert "PRESET = '128px'" in pushed and k.PRESET_LINE not in pushed
    assert k.EXPECT_ON in pushed and k.EXPECT_OFF not in pushed
    assert 'KAGGLE_TIME_BUDGET_HOURS = 6.0' in pushed and k.BUDGET_LINE not in pushed


def test_accounts_never_overwrite_each_other(tmp_path, monkeypatch):
    # Each account's kernel numbers its versions from 1; their outputs must not collide.
    e = k.kernel('canet_b4', '128px')
    assert k.output_dir(e, 'alice', 1) != k.output_dir(e, 'bob', 1)
    assert k.version_file(e, 'alice') != k.version_file(e, 'bob')
    cfg = tmp_path / 'kaggle.json'
    cfg.write_text('{"username": "someone", "key": "x"}')
    monkeypatch.setenv('KAGGLE_CONFIG_DIR', str(tmp_path))
    assert k.username() == 'someone'  # the CLI's own precedence, so pushes and fetches agree


@pytest.mark.parametrize(
    'text, state',
    [
        ('u/slug has status "KernelWorkerStatus.COMPLETE"', 'complete'),
        ('u/slug has status "KernelWorkerStatus.RUNNING"', 'running'),
        ('u/slug has status "KernelWorkerStatus.ERROR"', 'error'),
        ('u/slug has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', 'cancelled'),
        ('u/slug has status "QUEUED"', 'queued'),
        # A failed CLI call mentions "error" but says nothing about the kernel.
        ('ConnectionError: HTTPSConnectionPool: Max retries exceeded', 'unknown'),
    ],
)
def test_status_is_parsed_not_guessed(text, state):
    assert k.parse_status(text) == state


def test_cli_runs_in_utf8_mode(tmp_path, monkeypatch):
    # The CLI saves the kernel log with a bare open(path, 'w'); in a cp1252 locale the log's
    # non-ASCII characters raised UnicodeEncodeError and left the file empty. Do the same write
    # in the CLI's place, with the arguments kaggle() gives the CLI.
    log = tmp_path / 'kernel.log'
    script = f"import sys; open({str(log)!r}, 'w').write('≈ ±'); print(sys.flags.utf8_mode, '±')"
    real_run = k.subprocess.run
    monkeypatch.setattr(
        k.subprocess, 'run', lambda cmd, **kw: real_run([sys.executable, '-c', script], **kw)
    )
    assert k.kaggle('kernels', 'output') == '1 ±'
    assert log.read_text(encoding='utf-8') == '≈ ±'


def test_a_refused_push_stops_the_runner(tmp_path, monkeypatch):
    # An account runs at most two GPU sessions; the CLI reports a third push as an error but
    # exits 0, and the runner then watched a kernel that never started.
    e = dict(k.kernel('unet', '512px'), push_dir=str(tmp_path))
    refused = 'Kernel push error: Maximum batch GPU session count of 2 reached.'
    monkeypatch.setattr(k, 'kaggle', lambda *args, **kw: refused)
    with pytest.raises(SystemExit, match='Maximum batch GPU session count'):
        k.push(e, 'u', resume=False)
    assert not os.path.exists(k.version_file(e, 'u'))
    monkeypatch.setattr(k, 'kaggle', lambda *args, **kw: 'Kernel version 3 successfully pushed.')
    k.push(e, 'u', resume=False)
    with open(k.version_file(e, 'u')) as f:
        assert f.read() == '3'


def test_downloads_exclude_weights():
    files = ['runs/log.txt', 'runs/unet/seed42/best.pt', 'runs/unet/seed42/last.pt', 'x.pth']
    assert [f for f in files if re.match(k.NO_WEIGHTS, f)] == ['runs/log.txt']
