"""Checks that the patched dataset is cut from the source photographs as documented.

The script that made the Kaggle dataset `abdur548/nwrd-patched` was not kept, so its recipe was
reconstructed from the data (docs/protocol.md §3.2): each photograph is centred on a black canvas
of whole 512px patches (6144x4096 for the 6016x4000 photographs here) and cut, without resizing,
into non-overlapping 512x512 patches named <photo>_patch_<n>_x<left>_y<top>.png, numbered row by
row. Image patches are lossless crops of the decoded JPEG; mask patches keep the source masks'
grey values.

    python scripts/verify_dataset.py --source NWRDF --counts results/512px/final
    python scripts/verify_dataset.py --source NWRDF --patches wheat_rust_patches [--photos 91]

--source is the `NWRDF` folder of the public Kaggle dataset `manan369/nwrd-f` (the project's
private copy, `abdur548/nwrdf-unpatched`, is identical): <split>/images/<photo>.JPG and
<split>/masks/<photo>.png. With --counts only its masks are needed:
the rust-pixel count of every patch, saved with the 512px results (_cache/mask_stats_*.json and a
test_counts.npz), is recomputed from them. With --patches every patch file is compared pixel for
pixel with its crop of the padded source. Exits with 1 if anything differs.
"""

import argparse
import collections
import dataclasses
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

PATCH = 512
SPLITS = ('train', 'val', 'test')
NAME = re.compile(r'^(\d+)_patch_(\d+)_x(\d+)_y(\d+)\.png$')


@dataclasses.dataclass
class Report:
    checked: int = 0
    problems: list = dataclasses.field(default_factory=list)
    sizes: collections.Counter = dataclasses.field(default_factory=collections.Counter)


def pad_centred(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """Centres an image on a black canvas whose sides are whole multiples of `patch`."""
    h, w = img.shape[:2]
    H, W = -(-h // patch) * patch, -(-w // patch) * patch
    top, left = (H - h) // 2, (W - w) // 2
    return cv2.copyMakeBorder(
        img, top, H - h - top, left, W - w - left, cv2.BORDER_CONSTANT, value=0
    )


def grid(photo: int, shape: tuple, patch: int = PATCH) -> dict[str, tuple[int, int]]:
    """Patch file name -> (x, y) of its top-left corner, for every patch of a padded canvas."""
    cols = shape[1] // patch
    return {
        f'{photo}_patch_{r * cols + c:04d}_x{c * patch}_y{r * patch}.png': (c * patch, r * patch)
        for r in range(shape[0] // patch)
        for c in range(cols)
    }


def source_photos(source: str) -> dict[int, dict]:
    """photo -> {'split', 'image', 'mask'} of the source upload; a missing file is None."""
    photos = {}
    for split in SPLITS:
        for kind in ('images', 'masks'):
            for path in glob.glob(os.path.join(source, split, kind, '*')):
                stem = os.path.splitext(os.path.basename(path))[0]
                if not stem.isdigit():
                    continue
                entry = photos.setdefault(int(stem), {'split': split, 'image': None, 'mask': None})
                if entry['split'] != split:
                    raise ValueError(f'photograph {stem} is in both {entry["split"]} and {split}')
                entry['image' if kind == 'images' else 'mask'] = path
    return photos


def _read(path: str, flag: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    img = cv2.imread(path, flag)
    if img is None:
        raise OSError(f'cannot read {path}')
    return img


def saved_counts(run_dir: str) -> dict[str, tuple[str, int]]:
    """Patch name -> (split, rust pixels), as saved with a 512px run."""
    counts = {}
    for split in ('train', 'val'):
        path = os.path.join(run_dir, '_cache', f'mask_stats_{split}.json')
        with open(path, encoding='utf-8') as f:
            counts.update({name: (split, s['pos']) for name, s in json.load(f).items()})
    found = sorted(glob.glob(os.path.join(run_dir, '*', 'seed*', 'test_counts.npz')))
    if not found:
        raise FileNotFoundError(f'no test_counts.npz under {run_dir}')
    with np.load(found[0]) as z:  # columns tp, fp, fn, tn: the rust pixels are tp + fn
        rows = zip(z['files'], z['at_05'], strict=True)
        counts.update({str(f): ('test', int(c[0] + c[2])) for f, c in rows})
    return counts


def check_counts(source: str, run_dir: str, patch: int = PATCH) -> Report:
    """Recomputes every saved patch's rust-pixel count from the padded source masks."""
    photos, report = source_photos(source), Report()
    saved = collections.defaultdict(dict)
    for name, value in saved_counts(run_dir).items():
        m = NAME.match(name)
        if m:
            saved[int(m[1])][name] = value
        else:
            report.problems.append(f'unexpected patch name {name}')
    for photo, counts in sorted(saved.items()):
        src = photos.get(photo)
        if not src or not src['mask']:
            report.problems.append(f'photograph {photo}: no source mask')
            continue
        splits = sorted({split for split, _ in counts.values()})
        if splits != [src['split']]:
            report.problems.append(f'photograph {photo}: saved in {splits}, source {src["split"]}')
        mask = _read(src['mask'], cv2.IMREAD_GRAYSCALE)
        report.sizes[f'{mask.shape[1]}x{mask.shape[0]}'] += 1
        rust = pad_centred((mask > 127).astype(np.uint8), patch)
        cells = grid(photo, rust.shape, patch)
        if set(counts) != set(cells):
            report.problems.append(
                f'photograph {photo}: {len(counts)} patches saved, {len(cells)} expected'
            )
        for name in sorted(set(counts) & set(cells)):
            x, y = cells[name]
            pos = int(rust[y : y + patch, x : x + patch].sum())
            report.checked += 1
            if pos != counts[name][1]:
                report.problems.append(f'{name}: {counts[name][1]} rust pixels saved, {pos} now')
    missing = sorted(set(photos) - set(saved))
    if missing:
        report.problems.append(f'source photographs without saved patches: {missing}')
    return report


def check_patches(source: str, patches: str, only=(), patch: int = PATCH) -> Report:
    """Compares every patch file, pixel for pixel, with its crop of the padded source."""
    photos, report = source_photos(source), Report()
    for photo, src in sorted(photos.items()):
        if only and photo not in only:
            continue
        if not src['mask']:
            report.problems.append(f'photograph {photo}: no source mask')
            continue
        mask = _read(src['mask'])
        report.sizes[f'{mask.shape[1]}x{mask.shape[0]}'] += 1
        canvases = {'masks': pad_centred(mask, patch)}
        if not src['image']:
            report.problems.append(f'photograph {photo}: no source image')
        else:
            image = _read(src['image'], cv2.IMREAD_COLOR)
            if image.shape[:2] == mask.shape[:2]:
                canvases['images'] = pad_centred(image, patch)
            else:
                report.problems.append(f'photograph {photo}: image and mask sizes differ')
        cells = grid(photo, canvases['masks'].shape, patch)
        for kind, canvas in canvases.items():
            folder = os.path.join(patches, src['split'], kind)
            found = glob.glob(os.path.join(folder, f'{photo}_patch_*'))
            present = {os.path.basename(p) for p in found}
            for name in sorted(present - set(cells)):
                report.problems.append(f'unexpected {src["split"]}/{kind}/{name}')
            for name, (x, y) in cells.items():
                where = f'{src["split"]}/{kind}/{name}'
                if name not in present:
                    report.problems.append(f'missing {where}')
                    continue
                report.checked += 1
                crop = canvas[y : y + patch, x : x + patch]
                if not np.array_equal(_read(os.path.join(folder, name)), crop):
                    report.problems.append(f'{where} differs from the padded source')
    if only and not set(only) & set(photos):
        report.problems.append(f'none of the photographs {sorted(only)} is in the source')
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--source', required=True, help='the source upload (NWRDF)')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--counts', metavar='RUN', help='a 512px results folder (saved counts)')
    mode.add_argument('--patches', metavar='DIR', help='the patched dataset (wheat_rust_patches)')
    ap.add_argument('--photos', type=int, nargs='+', help='with --patches: check only these')
    a = ap.parse_args(argv)
    split_of = collections.Counter(p['split'] for p in source_photos(a.source).values())
    print('source:', ', '.join(f'{s} {split_of[s]}' for s in SPLITS), 'photographs')
    if a.counts:
        report, what = check_counts(a.source, a.counts), 'patch rust-pixel counts'
    else:
        report, what = check_patches(a.source, a.patches, set(a.photos or ())), 'patch files'
    print('photograph sizes:', ', '.join(f'{n} at {s}' for s, n in report.sizes.most_common()))
    print(f'{report.checked} {what} checked, {len(report.problems)} problems')
    for p in report.problems[:20]:
        print('  ' + p)
    if len(report.problems) > 20:
        print(f'  ... and {len(report.problems) - 20} more')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
