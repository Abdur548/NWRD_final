"""The data fetcher's safe extraction and manifest, and collecting a kernel's output."""

import json
import os
import sys
import zipfile

import pytest

from nwrd_public.dataset import load_manifest, verify_release

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import collect_results  # noqa: E402
import fetch_data  # noqa: E402

LAYOUT = {'train/images': 1, 'train/masks': 1}


def _zip(path, names):
    with zipfile.ZipFile(path, 'w') as z:
        for n in names:
            z.writestr(n, b'pixels of ' + n.encode())
    return str(path)


@pytest.mark.parametrize(
    'names, problem',
    [
        (['train/images/1.jpg', '../masks/1.png'], 'unexpected entry'),
        (['train/images/1.jpg', 'train/masks/1.py'], 'unexpected entry'),
        (['train/images/1.jpg', 'C:/train/masks/1.png'], 'unexpected entry'),
        (['train/images/1.jpg', 'train/images/2.jpg'], 'unexpected layout'),
    ],
)
def test_unsafe_or_unexpected_archives_are_not_extracted(tmp_path, names, problem):
    with pytest.raises(SystemExit, match=problem):
        fetch_data.safe_extract(_zip(tmp_path / 'a.zip', names), str(tmp_path / 'out'), LAYOUT)
    assert not (tmp_path / 'out').exists()


def test_extraction_and_manifest(tmp_path):
    archive = _zip(tmp_path / 'a.zip', ['train/images/1.jpg', 'train/masks/1.png'])
    out, path = tmp_path / 'out', str(tmp_path / 'manifest.json')
    assert fetch_data.safe_extract(archive, str(out), LAYOUT) == LAYOUT
    (out / 'train' / 'images' / 'Thumbs.db').write_bytes(b'left by Windows')
    fetch_data.write_manifest(str(out), path, LAYOUT)
    manifest = load_manifest(path)
    assert sorted(manifest['files']) == ['train/images/1.jpg', 'train/masks/1.png']
    assert 'zenodo.21672928' in manifest['source']
    verify_release(str(out), manifest)
    (out / 'train' / 'images' / '2.jpg').write_bytes(b'one photograph too many')
    with pytest.raises(SystemExit, match='holds 2 .jpg files, expected 1'):
        fetch_data.write_manifest(str(out), path, LAYOUT)


def test_manifest_is_written_only_from_a_fresh_extraction(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_data, 'check_archive', lambda path: None)
    (tmp_path / 'raw' / 'NWRD').mkdir(parents=True)
    (tmp_path / 'raw' / 'NWRD' / 'edited.jpg').write_bytes(b'not from the archive')
    archive = _zip(tmp_path / 'a.zip', ['train/images/1.jpg'])
    args = ['--zip', archive, '--dest', str(tmp_path / 'raw'), '--write-manifest',
            '--manifest', str(tmp_path / 'manifest.json')]  # fmt: skip
    with pytest.raises(SystemExit, match='fresh extraction'):
        fetch_data.main(args)
    assert not (tmp_path / 'manifest.json').exists()


def _kernel_output(tmp_path, config: dict, runs: dict) -> str:
    """A kernel's downloaded output: runs/config.json and one metrics.json per finished run."""
    root = tmp_path / 'kernel' / 'runs'
    for name, metrics in runs.items():
        (root / config['model'] / name).mkdir(parents=True)
        if metrics is not None:
            (root / config['model'] / name / 'metrics.json').write_text(json.dumps(metrics))
    (root / 'config.json').write_text(json.dumps(config))
    return str(tmp_path / 'kernel')


def test_unfinished_kernels_are_not_collected(tmp_path):
    cfg = {'model': 'unet', 'window': 128, 'scale': 0.5}
    src = _kernel_output(tmp_path, cfg, {'seed42': cfg, 'seed43': None})
    with pytest.raises(SystemExit, match='1 of 2 runs'):
        collect_results.collect(src, str(tmp_path / 'results'))
    assert not (tmp_path / 'results').exists()


def test_runs_of_another_scale_are_not_collected(tmp_path):
    cfg = {'model': 'unet', 'window': 128, 'scale': 0.5}
    src = _kernel_output(tmp_path, cfg, {'seed42': {**cfg, 'window': 512, 'scale': 1.0}})
    with pytest.raises(SystemExit, match='not collecting'):
        collect_results.collect(src, str(tmp_path / 'results'))
    assert not (tmp_path / 'results').exists()
