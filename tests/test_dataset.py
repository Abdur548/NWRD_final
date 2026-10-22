"""The release: the split, the authors' validation draw, the MD5 check, finding the release on
Kaggle, and reading photographs and masks."""

import os
import shutil

import cv2
import numpy as np
import pytest

from nwrd_public.dataset import (
    EXCLUDED_MISALIGNED,
    EXCLUDED_TWINS,
    PAPER_VAL_IDS,
    TWIN_PAIRS,
    VAL_IDS,
    camera,
    find_release,
    load_manifest,
    md5_of,
    open_release,
    paper_validation_ids,
    read_photo,
    splits,
    verify_release,
)


def _quiet(_msg):
    pass


def test_authors_validation_draw_gives_the_documented_ids():
    train_images = [n for n in load_manifest()['files'] if n.startswith('train/images/')]
    assert len(train_images) == 90
    assert sorted(paper_validation_ids(train_images), key=int) == list(PAPER_VAL_IDS)


def test_split_is_the_authors_minus_the_twins_and_101():
    s = splits(load_manifest())
    assert [len(s[k]) for k in ('train', 'val', 'test')] == [77, 8, 10]
    assert s['val'] == list(VAL_IDS) == [i for i in PAPER_VAL_IDS if i != '101']
    assert s['test'] == ['25', '29', '62', '63', '67', '77', '80', '89', '105', '123']
    used = set(s['train']) | set(s['val']) | set(s['test'])
    assert not (set(EXCLUDED_TWINS) | set(EXCLUDED_MISALIGNED)) & used
    assert len(used) == 95


def test_no_val_or_test_photo_has_a_twin_in_training():
    s = splits(load_manifest())
    train, held_out = set(s['train']), set(s['val'] + s['test'])
    for pair in TWIN_PAIRS:
        assert not (set(pair) & train and set(pair) & held_out), pair
    # The excluded photographs are exactly the twins of validation and test photographs.
    twins_of_held_out = {x for p in TWIN_PAIRS if set(p) & held_out for x in p} - held_out
    assert twins_of_held_out == set(EXCLUDED_TWINS)
    assert set(EXCLUDED_TWINS.values()) <= held_out


def _tiny(tmp_path):
    """Two files and their manifest."""
    root = tmp_path / 'release'
    for rel, data in (('test/images/105.jpg', b'photo'), ('test/masks/105.png', b'mask')):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    files = {
        rel: {'md5': md5_of(str(root / rel)), 'bytes': (root / rel).stat().st_size}
        for rel in ('test/images/105.jpg', 'test/masks/105.png')
    }
    return root, {'files': files}


def test_changed_or_missing_file_stops_the_run(tmp_path):
    root, manifest = _tiny(tmp_path)
    (root / 'README.md').write_text('a file the manifest does not list')
    verify_release(str(root), manifest)  # extra files are ignored
    (root / 'test' / 'images' / '105.jpg').write_bytes(b'phot0')  # same size, other bytes
    with pytest.raises(RuntimeError, match='1 changed'):
        verify_release(str(root), manifest)
    (root / 'test' / 'masks' / '105.png').unlink()
    with pytest.raises(RuntimeError, match='1 missing'):
        verify_release(str(root), manifest)


def test_release_is_found_when_nested(tmp_path):
    root, manifest = _tiny(tmp_path)
    assert find_release(str(root), manifest) == str(root)
    nested = tmp_path / 'input' / 'nwrd-wheat' / 'v1'
    shutil.copytree(root, nested)
    assert os.path.samefile(find_release(str(tmp_path / 'input'), manifest), nested)
    with pytest.raises(FileNotFoundError, match='bcap52/nwrd-wheat'):
        find_release(str(tmp_path / 'input' / 'nwrd-wheat' / 'v1' / 'test' / 'masks'), manifest)


def test_open_release_checks_splits_and_limits(synthetic_release):
    r = open_release(synthetic_release.root, synthetic_release.manifest, limit=2, log=_quiet)
    assert r.root == synthetic_release.root
    assert r.splits == {'train': ['3', '5'], 'val': ['2', '30'], 'test': ['25', '29']}
    assert r.folders['2'] == 'train' and r.folders['105'] == 'test' and len(r.folders) == 100


def test_a_mask_of_another_size_is_an_error(synthetic_release, tmp_path):
    root = synthetic_release.root
    rgb, rust = read_photo(root, 'train', '10')  # mask and photograph of the same size
    full = cv2.cvtColor(
        cv2.imread(os.path.join(root, 'train', 'images', '10.jpg')), cv2.COLOR_BGR2RGB
    )
    assert np.array_equal(rgb, full) and rust.shape == full.shape[:2] and rust.dtype == bool
    assert rust.any()
    # 101's smaller mask cannot be placed by assumption: in the release it fits only shifted past
    # the photograph's edge, so 101 is in no split.
    with pytest.raises(ValueError, match='does not match the photograph'):
        read_photo(root, 'train', '101')
    copy = tmp_path / 'copy'
    shutil.copytree(root, copy)
    cv2.imwrite(
        str(copy / 'train' / 'masks' / '10.png'),
        np.zeros((full.shape[0] + 1, full.shape[1]), np.uint8),
    )
    with pytest.raises(ValueError, match='does not match the photograph'):
        read_photo(str(copy), 'train', '10')


def test_camera_reads_the_exif_model(tmp_path):
    from PIL import Image

    (tmp_path / 'test' / 'images').mkdir(parents=True)
    exif = Image.Exif()
    exif[272] = 'NIKON D3200'
    Image.new('RGB', (8, 8)).save(tmp_path / 'test' / 'images' / '1.jpg', exif=exif)
    Image.new('RGB', (8, 8)).save(tmp_path / 'test' / 'images' / '2.jpg')
    assert camera(str(tmp_path), 'test', '1') == 'NIKON D3200'
    assert camera(str(tmp_path), 'test', '2') == 'no EXIF'
