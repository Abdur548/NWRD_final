"""The real release in data/raw/NWRD: its files, its masks, and the window counts the docs quote.
Skipped when the release has not been fetched (python scripts/fetch_data.py)."""

import os

import pytest
from PIL import Image

from nwrd_public.dataset import load_manifest, photo_folders, read_mask, splits, verify_release
from nwrd_public.evaluate import paper_order
from nwrd_public.windows import grid, paper_grid, resize_mask, scaled_shape, window_rust

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
    masks = {p: read_mask(real_release, folders[p], p) for p in s['train'] + s['val'] + s['test']}
    expected = {  # results/dataset/summary.md, "Our split (D3): windows per split"
        ('train', 512): (5626, 1349),
        ('val', 512): (619, 134),
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
    half = {p: scaled_shape(*masks[p].shape, 0.5) for p in s['test'] + s['val']}
    assert sum(len(paper_grid(*half[p])) for p in test) == 2737
    assert sum(len(paper_grid(*half[p])) for p in s['val']) == 2237
    assert len(grid(*masks['101'].shape, 512)) == 56
