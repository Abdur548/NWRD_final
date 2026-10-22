"""Shared fixtures: a miniature NWRD release with every file name of the real one, small synthetic
photographs of mixed sizes, and its own manifest; and the real release when it is present."""

import json
import os
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from nwrd_public.dataset import md5_of, photo_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, 'data', 'manifest.json')
REAL_RELEASE = os.path.join(ROOT, 'data', 'raw', 'NWRD')
# (height, width), in turn; every side stays at least 128 px when halved
SIZES = [(280, 300), (300, 280), (288, 384), (410, 296)]


def write_release(root: str) -> str:
    """Writes the miniature release under `root` and returns its manifest's path.

    Every photograph has one rust square near its top left, and every third a second one near
    its bottom right; photo 101's mask is 19 px narrower and 21 px shorter than its photograph,
    as in the release (it is in no split, and reading it is an error)."""
    with open(MANIFEST, encoding='utf-8') as f:
        names = json.load(f)['files']
    ids = sorted({photo_id(n) for n in names}, key=int)
    rng = np.random.default_rng(0)
    files = {}
    for k, pid in enumerate(ids):
        folder = 'test' if f'test/images/{pid}.jpg' in names else 'train'
        h, w = SIZES[k % len(SIZES)]
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), np.uint8)
        mask[h // 8 : h // 8 + 24, w // 8 : w // 8 + 24] = 255
        if k % 3 == 0:
            mask[-60:-40, -60:-40] = 255
        img[mask > 0] = (30, 120, 200)  # BGR orange
        if pid == '101':
            mask = mask[: h - 21, : w - 19]
        for kind, ext, arr in (('images', 'jpg', img), ('masks', 'png', mask)):
            rel = f'{folder}/{kind}/{pid}.{ext}'
            path = os.path.join(root, *rel.split('/'))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, arr)
            files[rel] = {'md5': md5_of(path), 'bytes': os.path.getsize(path)}
    manifest = os.path.join(root, 'manifest.json')
    with open(manifest, 'w', encoding='utf-8') as f:
        json.dump({'source': 'synthetic', 'files': files}, f)
    return manifest


@pytest.fixture(scope='session')
def synthetic_release(tmp_path_factory):
    """SimpleNamespace(root, manifest): the miniature release, built once per session. Treat it
    as read-only."""
    root = str(tmp_path_factory.mktemp('release'))
    return SimpleNamespace(root=root, manifest=write_release(root))


@pytest.fixture
def real_release():
    if not os.path.isdir(os.path.join(REAL_RELEASE, 'train')):
        pytest.skip('the release is not in data/raw/NWRD (python scripts/fetch_data.py)')
    return REAL_RELEASE
