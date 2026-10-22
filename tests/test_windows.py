"""Fitted-stride windows: every pixel covered without padding and scored exactly once; resizing;
rust counts; the paper's patch grid; the training sample and its cut."""

import math
import os
from itertools import pairwise

import cv2
import numpy as np
import pytest

from nwrd_public.dataset import open_release, read_photo
from nwrd_public.windows import (
    axis_starts,
    cut_training_windows,
    grid,
    owned_spans,
    paper_grid,
    pos_weight,
    resize_mask,
    resize_photo,
    scaled_shape,
    select_windows,
    window_name,
    window_rust,
)

# Every side length in the release (docs/dataset.md): the three cameras' sizes and the crops.
SIDES = [4608, 6016, 4000, 3456, 2536, 4600, 2909, 2984, 4248, 3584, 3593, 3435, 3424, 3968, 4604,
         2844]  # fmt: skip


def _quiet(_msg):
    pass


def overlap(length: int, size: int = 512) -> int:
    s = axis_starts(length, size)
    return max((a + size - b for a, b in pairwise(s)), default=0)


@pytest.mark.parametrize('length', SIDES)
@pytest.mark.parametrize('size, scale', [(512, 1.0), (128, 0.5)])
def test_windows_cover_every_pixel_without_padding(length, size, scale):
    side = round(length * scale)
    s = axis_starts(side, size)
    assert len(s) == math.ceil(side / size) and s[0] == 0 and s[-1] == side - size
    covered, owned = np.zeros(side, int), np.zeros(side, int)
    spans = owned_spans(s, size, side)
    for a, (lo, hi) in zip(s, spans, strict=True):
        covered[a : a + size] += 1
        owned[lo:hi] += 1
        assert a <= lo < hi <= a + size  # a window scores only pixels it covers
    assert covered.min() >= 1 and (owned == 1).all()
    for (a, b), (_, boundary) in zip(pairwise(s), spans[:-1], strict=True):  # split at midpoints
        assert abs(boundary - (a + b + size) / 2) <= 0.5


def test_documented_overlaps():
    assert [overlap(n) for n in (4608, 6016, 4000, 3456, 3593)] == [0, 12, 14, 22, 72]
    assert max(overlap(round(n / 2), 128) for n in SIDES) == 11


def test_short_sides():
    assert axis_starts(512, 512) == [0] and owned_spans([0], 512, 512) == [(0, 512)]
    with pytest.raises(ValueError, match='never padded'):
        axis_starts(500, 512)


def test_grid_is_row_major():
    assert grid(600, 1100, 512) == [(0, 0), (0, 294), (0, 588), (88, 0), (88, 294), (88, 588)]


def test_resizing_photo_and_mask_agree():
    assert scaled_shape(2909, 4608, 0.5) == (1454, 2304)
    assert scaled_shape(3435, 3593, 0.5) == (1718, 1796)
    photo = np.random.default_rng(0).integers(0, 256, (7, 9, 3), dtype=np.uint8)
    assert resize_photo(photo, 1.0) is photo and resize_photo(photo, 0.5).shape == (4, 4, 3)
    mask = np.zeros((7, 9), bool)
    assert resize_mask(mask, 1.0) is mask
    assert resize_mask(mask, 0.5).shape == (4, 4) and resize_mask(mask, 0.5).dtype == bool


def test_half_size_mask_is_rust_where_half_the_area_is():
    m = np.zeros((4, 4), bool)
    m[0, 0] = m[0, 1] = True  # 2 of 4 pixels
    m[2, 2] = True  # 1 of 4
    assert resize_mask(m, 0.5).tolist() == [[True, False], [False, False]]


def test_window_rust_matches_direct_sums():
    m = np.random.default_rng(0).random((300, 700)) > 0.97
    positions, counts = window_rust(m, 128)
    assert positions == grid(300, 700, 128)
    assert counts.tolist() == [int(m[y : y + 128, x : x + 128].sum()) for y, x in positions]


def test_paper_grid_drops_the_remainder_row_major():
    assert paper_grid(300, 260) == [(0, 0), (0, 128), (128, 0), (128, 128)]
    assert len(paper_grid(2000, 3008)) == 15 * 23  # a halved Nikon photograph


def test_selection_keeps_every_rust_window_and_a_seeded_fifth_of_the_rest():
    stats = {window_name(str(p), 0, x): {'pos': int(p < 10), 'total': 100}
             for p in range(110) for x in (0,)}  # fmt: skip
    files, info = select_windows(stats, 0.2, seed=42)
    assert info == {'rust_windows': 10, 'bg_kept': 20, 'bg_total': 100} and len(files) == 30
    assert files == sorted(files) and files == select_windows(stats, 0.2, seed=42)[0]
    assert files != select_windows(stats, 0.2, seed=43)[0]
    assert pos_weight(stats, files) == (3000 - 10) / 10


def test_window_names_sort_by_photograph_then_position():
    assert window_name('7', 0, 1001) == '7_y0000_x1001.png'


def test_cutting_is_deterministic_and_cached(synthetic_release, tmp_path):
    release = open_release(synthetic_release.root, synthetic_release.manifest, limit=4, log=_quiet)
    ids = release.splits['train']
    args = (128, 1.0, 0.2, 42)
    a = cut_training_windows(release, ids, str(tmp_path / 'a'), *args, workers=1, log=_quiet)
    b = cut_training_windows(release, ids, str(tmp_path / 'b'), *args, workers=2, log=_quiet)
    assert a['files'] == b['files'] and a['sample'] == len(a['files'])
    assert 0 < a['rust_windows'] < a['sample'] < a['windows']
    assert sorted(os.listdir(tmp_path / 'a' / 'images')) == a['files']
    for f in a['files']:
        for kind in ('images', 'masks'):
            assert (tmp_path / 'a' / kind / f).read_bytes() == (
                tmp_path / 'b' / kind / f
            ).read_bytes()

    logs = []
    assert cut_training_windows(release, ids, str(tmp_path / 'a'), *args, log=logs.append) == a
    assert any('reusing' in m for m in logs)
    with pytest.raises(RuntimeError, match='other settings'):
        cut_training_windows(release, ids, str(tmp_path / 'a'), 128, 1.0, 0.5, 42, log=_quiet)


def test_a_failed_write_is_never_cached(synthetic_release, tmp_path, monkeypatch):
    import nwrd_public.windows as w

    release = open_release(synthetic_release.root, synthetic_release.manifest, limit=2, log=_quiet)
    monkeypatch.setattr(w.cv2, 'imwrite', lambda path, arr: False)  # as on a full disk
    with pytest.raises(OSError, match='could not write'):
        cut_training_windows(release, ['3'], str(tmp_path), 128, 1.0, 0.2, 42, log=_quiet)
    assert not (tmp_path / 'windows.json').exists()


def test_halving_removes_detail_half_the_resolution_cannot_hold():
    # A fine stripe pattern (0.45 cycles per pixel) cannot survive halving; an aliasing resize
    # would fold it back as a coarse false pattern.
    x = np.arange(256)
    stripes = (127.5 + 100 * np.cos(2 * np.pi * 0.45 * x))[None, :, None].repeat(64, 0)
    half = resize_photo(np.repeat(stripes, 3, 2).astype(np.uint8), 0.5).astype(float)
    assert np.abs(half[8:-8, 8:-8] - half[8:-8, 8:-8].mean()).max() < 5


@pytest.mark.parametrize('scale', [1.0, 0.5])
def test_cut_windows_are_crops_of_the_resized_photograph(synthetic_release, tmp_path, scale):
    release = open_release(synthetic_release.root, synthetic_release.manifest, limit=2, log=_quiet)
    info = cut_training_windows(release, ['101'], str(tmp_path), 128, scale, 1.0, 42, log=_quiet)
    rgb, rust = read_photo(release.root, 'train', '101')
    rgb, rust = resize_photo(rgb, scale), resize_mask(rust, scale)
    assert info['windows'] == len(grid(*rust.shape, 128)) == info['sample']  # every window kept
    for (y, x), name in zip(grid(*rust.shape, 128), info['files'], strict=True):
        assert name == window_name('101', y, x)
        img = cv2.cvtColor(cv2.imread(str(tmp_path / 'images' / name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(tmp_path / 'masks' / name), cv2.IMREAD_GRAYSCALE) > 127
        assert np.array_equal(img, rgb[y : y + 128, x : x + 128])
        assert np.array_equal(mask, rust[y : y + 128, x : x + 128])
