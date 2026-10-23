"""scripts/verify_dataset.py: the dataset's patching recipe, checked on synthetic photographs."""

import json
import os
import sys

import cv2
import numpy as np
import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import verify_dataset as v  # noqa: E402

P = 64  # small patches keep the synthetic photographs small
PHOTOS = {'train': 1, 'val': 2, 'test': 3}


@pytest.fixture
def source(tmp_path):
    """One 150x100 photograph per split with a greyscale mask: a 3x2 grid of 64px patches."""
    rng = np.random.default_rng(0)
    for split, photo in PHOTOS.items():
        for kind in ('images', 'masks'):
            (tmp_path / 'src' / split / kind).mkdir(parents=True)
        mask = np.zeros((100, 150), np.uint8)
        mask[20 + photo : 70, 30:120] = 255
        mask[70, 30:120] = 128  # a soft edge, as the real masks have
        image = rng.integers(0, 256, (100, 150, 3), np.uint8)
        cv2.imwrite(str(tmp_path / 'src' / split / 'images' / f'{photo}.JPG'), image)
        cv2.imwrite(str(tmp_path / 'src' / split / 'masks' / f'{photo}.png'), mask)
    return tmp_path / 'src'


def cut(source, out, patch=P, centred=True):
    """Cuts the source as documented, independently of the script: a centred black canvas."""
    for split, photo in PHOTOS.items():
        for kind, ext in (('images', 'JPG'), ('masks', 'png')):
            img = cv2.imread(str(source / split / kind / f'{photo}.{ext}'), cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            H, W = -(-h // patch) * patch, -(-w // patch) * patch
            y0, x0 = ((H - h) // 2, (W - w) // 2) if centred else (0, 0)
            canvas = np.zeros((H, W, *img.shape[2:]), np.uint8)
            canvas[y0 : y0 + h, x0 : x0 + w] = img
            (out / split / kind).mkdir(parents=True, exist_ok=True)
            for r in range(H // patch):
                for c in range(W // patch):
                    name = f'{photo}_patch_{r * (W // patch) + c:04d}_x{c * patch}_y{r * patch}.png'
                    tile = canvas[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch]
                    cv2.imwrite(str(out / split / kind / name), tile)


def test_the_documented_canvas_of_a_6016x4000_photograph():
    canvas = v.pad_centred(np.ones((4000, 6016), np.uint8))
    assert canvas.shape == (4096, 6144) and canvas.sum() == 4000 * 6016
    assert canvas[48:4048, 64:6080].all()  # 48 px black at top and bottom, 64 at left and right
    cells = v.grid(91, canvas.shape)
    assert len(cells) == 96 and cells['91_patch_0095_x5632_y3584.png'] == (5632, 3584)


def test_patches_cut_as_documented_pass(source, tmp_path):
    cut(source, tmp_path / 'patches')
    report = v.check_patches(str(source), str(tmp_path / 'patches'), patch=P)
    assert report.problems == [] and report.checked == 3 * 6 * 2
    assert report.sizes == {'150x100': 3}


def test_changed_and_missing_patches_and_other_padding_are_reported(source, tmp_path):
    out = tmp_path / 'patches'
    cut(source, out)
    edited = out / 'test' / 'masks' / '3_patch_0004_x64_y64.png'
    mask = cv2.imread(str(edited), cv2.IMREAD_UNCHANGED)
    mask[0, 0] ^= 1
    cv2.imwrite(str(edited), mask)
    (out / 'val' / 'images' / '2_patch_0000_x0_y0.png').unlink()
    assert v.check_patches(str(source), str(out), patch=P).problems == [
        'missing val/images/2_patch_0000_x0_y0.png',
        'test/masks/3_patch_0004_x64_y64.png differs from the padded source',
    ]
    cut(source, tmp_path / 'top_left', centred=False)
    problems = v.check_patches(str(source), str(tmp_path / 'top_left'), patch=P).problems
    assert problems and all(p.endswith('differs from the padded source') for p in problems)


def test_counts_mode_recomputes_the_saved_rust_pixels(source, tmp_path):
    cut(source, tmp_path / 'patches')
    pos = {}
    for split in PHOTOS:
        for f in sorted((tmp_path / 'patches' / split / 'masks').iterdir()):
            pos[f.name] = split, int((cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127).sum())
    run = tmp_path / 'run'
    (run / '_cache').mkdir(parents=True)
    for split in ('train', 'val'):
        stats = {n: {'pos': p, 'total': P * P} for n, (s, p) in pos.items() if s == split}
        (run / '_cache' / f'mask_stats_{split}.json').write_text(json.dumps(stats))
    test = sorted((n, p) for n, (s, p) in pos.items() if s == 'test')
    counts = np.array([[p // 2, 5, p - p // 2, P * P - p - 5] for _, p in test])  # tp fp fn tn
    (run / 'unet' / 'seed42').mkdir(parents=True)
    np.savez(
        run / 'unet' / 'seed42' / 'test_counts.npz',
        files=np.array([n for n, _ in test]),
        at_05=counts,
        at_thr=counts,
    )
    report = v.check_counts(str(source), str(run), patch=P)
    assert report.problems == [] and report.checked == 18

    stats_file = run / '_cache' / 'mask_stats_val.json'
    stats = json.loads(stats_file.read_text())
    stats['2_patch_0001_x64_y0.png']['pos'] += 1
    stats_file.write_text(json.dumps(stats))
    problems = v.check_counts(str(source), str(run), patch=P).problems
    assert len(problems) == 1 and problems[0].startswith('2_patch_0001_x64_y0.png: ')


def test_cli_exit_status(source, tmp_path, capsys):
    cut(source, tmp_path / 'patches', patch=v.PATCH)  # each photograph fits one 512px patch
    assert v.main(['--source', str(source), '--patches', str(tmp_path / 'patches')]) == 0
    assert '6 patch files checked, 0 problems' in capsys.readouterr().out
    assert v.main(['--source', str(source), '--patches', str(tmp_path), '--photos', '9']) == 1
