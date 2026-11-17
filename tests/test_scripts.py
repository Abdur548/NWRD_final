"""The data fetcher's safe extraction and manifest, collecting a kernel's output, the dataset
study's mask placement search, and the patching script."""

import csv
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import pytest

from nwrd_public.dataset import load_manifest, open_release, verify_release

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import collect_results  # noqa: E402
import describe_data  # noqa: E402
import fetch_data  # noqa: E402
import patch_release  # noqa: E402

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


def _orange_blobs(h: int, w: int, seed: int = 0):
    """A green photograph (RGB) with orange squares, and where they are."""
    rng = np.random.default_rng(seed)
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[...] = (40, 150, 40)
    rgb += rng.integers(0, 20, rgb.shape, dtype=np.uint8)
    blobs = np.zeros((h, w), bool)
    for y, x in rng.integers(0, (h - 6, w - 6), (25, 2)):
        blobs[y : y + 6, x : x + 6] = True
    rgb[blobs] = (200, 120, 30)
    return rgb, blobs


def test_placement_search_finds_a_mask_shifted_past_the_edge():
    # Photo 101's case: a smaller mask that fits only with its corner 1 px above and 8 px left of
    # the photograph's, which a search inside the photograph alone cannot find.
    rgb, blobs = _orange_blobs(100, 120)
    mask = np.zeros((79, 101), bool)
    mask[1:, 8:] = blobs[:78, :93]
    p = describe_data.placement_search(rgb, mask, margin=12)
    assert p['best_yx'] == [-1, -8] and p['best'] > p['top_left']
    assert {'centre', 'bottom_right', 'stretched'} <= set(p)
    # A mask of the photograph's size, in place: best where it is, nothing stretched.
    p = describe_data.placement_search(rgb, blobs, margin=12)
    assert p['best_yx'] == [0, 0] and p['best'] == p['top_left'] and 'stretched' not in p


def test_placement_figure_shows_both_placements(tmp_path):
    rgb, blobs = _orange_blobs(100, 120)
    mask = np.zeros((79, 101), bool)
    mask[1:, 8:] = blobs[:78, :93]
    path = tmp_path / 'mask.png'
    describe_data.placement_figure(rgb, mask, [-1, -8], str(path), crops=2, half=10, zoom=2)
    with describe_data.Image.open(path) as im:
        assert im.size == (3 * 40 + 2 * 6, 2 * 40 + 6)  # 2 rows of 3 panels, 6 px gaps


def _quiet(_msg):
    pass


def _index(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize('preset, size', [('512px', 512), ('128px', 128)])
def test_patches_tile_every_photograph_once(synthetic_release, tmp_path, preset, size):
    release = open_release(synthetic_release.root, synthetic_release.manifest, limit=2, log=_quiet)
    if preset == '512px':  # the miniature photographs are smaller than a 512px patch
        with pytest.raises(ValueError, match='never padded'):
            patch_release.patch(release, str(tmp_path), preset, log=_quiet)
        return
    rows = _index(patch_release.patch(release, str(tmp_path), preset, log=_quiet))
    assert {r['split'] for r in rows} == {'train', 'val', 'test'}
    for pid in {r['photo'] for r in rows}:
        mine = [r for r in rows if r['photo'] == pid]
        h = max(int(r['y']) for r in mine) + size  # the last patch ends at the photograph's edge
        w = max(int(r['x']) for r in mine) + size
        owned = np.zeros((h, w), int)
        for r in mine:
            y, x = int(r['y']), int(r['x'])
            y0, y1, x0, x1 = (int(r[k]) for k in ('own_y0', 'own_y1', 'own_x0', 'own_x1'))
            assert y <= y0 < y1 <= y + size and x <= x0 < x1 <= x + size
            owned[y0:y1, x0:x1] += 1
            mask = cv2.imread(str(tmp_path / r['file'].replace('/images/', '/masks/')), 0)
            assert mask.shape == (size, size) and int((mask > 127).sum()) == int(r['rust_px'])
            assert os.path.exists(tmp_path / r['file'])
        assert (owned == 1).all()  # stitched from the owned regions, no pixel counts twice


def test_sample_keeps_every_rust_patch_and_only_cuts_training(synthetic_release, tmp_path):
    release = open_release(synthetic_release.root, synthetic_release.manifest, limit=2, log=_quiet)
    every = _index(
        patch_release.patch(release, str(tmp_path / 'all'), '128px', ('train',), log=_quiet)
    )
    some = _index(
        patch_release.patch(release, str(tmp_path / 's'), '128px', ('train',), True, log=_quiet)
    )
    rust = {r['file'] for r in every if int(r['rust_px'])}
    assert rust <= {r['file'] for r in some} <= {r['file'] for r in every}
    assert {r['split'] for r in some} == {'train'}
