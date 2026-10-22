"""The real release in data/raw/NWRD: its files, its masks, and the window counts the docs quote.
Skipped when the release has not been fetched (python scripts/fetch_data.py)."""

import os
import sys

import numpy as np
import pytest
from PIL import Image

from nwrd_public.dataset import (
    PAPER_VAL_IDS,
    load_manifest,
    photo_folders,
    read_mask,
    splits,
    verify_release,
)
from nwrd_public.evaluate import paper_order
from nwrd_public.windows import paper_grid, resize_mask, scaled_shape, window_rust

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import describe_data  # noqa: E402

pytestmark = pytest.mark.data


def _size(root, folder, kind, pid):
    ext = 'jpg' if kind == 'images' else 'png'
    with Image.open(os.path.join(root, folder, kind, f'{pid}.{ext}')) as im:
        return im.size


def test_the_release_matches_the_manifest(real_release):
    verify_release(real_release, load_manifest())


def test_every_mask_has_its_photographs_size_except_101(real_release):
    folders = photo_folders(load_manifest())
    differ = {}
    for pid, folder in folders.items():
        photo = _size(real_release, folder, 'images', pid)
        mask = _size(real_release, folder, 'masks', pid)
        if photo != mask:
            differ[pid] = (photo, mask)
    assert differ == {'101': ((3612, 3456), (3593, 3435))}


def test_window_counts_of_the_split(real_release):
    manifest = load_manifest()
    folders, s = photo_folders(manifest), splits(manifest)
    used = s['train'] + s['val'] + s['test']
    masks = {p: read_mask(real_release, folders[p], p) for p in used + list(PAPER_VAL_IDS)}
    expected = {  # results/dataset/summary.md, "Our split (D3): windows per split"
        ('train', 512): (5626, 1349),
        ('val', 512): (563, 116),
        ('test', 512): (753, 232),
        ('train', 128): (22492, 2230),
    }
    for (split, size), (windows, rust) in expected.items():
        scale = 1.0 if size == 512 else 0.5
        counts = [window_rust(resize_mask(masks[p], scale), size)[1] for p in s[split]]
        assert (sum(len(c) for c in counts), sum(int((c > 0).sum()) for c in counts)) == (
            windows,
            rust,
        ), (split, size)
    # The paper's 2,737 test patches, in the authors' order, and its 2,237 validation patches.
    test = paper_order(s['test'])
    assert test == ['105', '123', '25', '29', '62', '63', '67', '77', '80', '89']
    half = {p: scaled_shape(*masks[p].shape, 0.5) for p in s['test'] + list(PAPER_VAL_IDS)}
    assert sum(len(paper_grid(*half[p])) for p in test) == 2737
    assert sum(len(paper_grid(*half[p])) for p in PAPER_VAL_IDS) == 2237  # 101 included


def test_mask_101_fits_only_shifted_past_the_edge(real_release):
    # docs/dataset.md §3: why 101 is in no split.
    with Image.open(os.path.join(real_release, 'train', 'images', '101.jpg')) as im:
        rgb = np.asarray(im.convert('RGB'))
    p = describe_data.placement_search(rgb, read_mask(real_release, 'train', '101'), margin=10)
    assert p['best_yx'] == [-1, -8] and p['best'] - p['top_left'] > 3
