"""Cuts the public NWRD release into patches on the pipeline's fitted-stride grid and writes them to
disk with an index, to inspect them or use them outside the Kaggle notebooks.

    python -P -E scripts/patch_release.py                          # 512px: every patch, every split
    python -P -E scripts/patch_release.py --preset 128px           # the paper's scale, halved
    python -P -E scripts/patch_release.py --splits train --sample  # the kernels' training sample

A side of L px gets ceil(L / P) patches of P px spread evenly from 0 to L - P: no padding, every
pixel covered, and neighbours overlapping by what L lacks of a whole number of patches (at most
46 px at 512px and 11 px at 128px; docs/protocol.md §3.1). The split is the study's: 77 / 8 / 10
photographs, with photo 101 and the four twins in none. The patches are cut by the code the
kernels run (windows.cut_training_windows), so they are what training reads, pixel for pixel.

Under --out (default build/patches/<preset>, on the repository's drive):
  <split>/images/<photo>_y<y>_x<x>.png  lossless RGB crops (of the halved photograph at 128px)
  <split>/masks/<photo>_y<y>_x<x>.png   rust masks, 0 or 255
  <split>/windows.json                  the cut's settings and counts
  patches.csv                           one row per patch: split, photo, file, y, x, size, scale,
                                        rust_px, and own_y0, own_y1, own_x0, own_x1, the pixels
                                        [own_y0, own_y1) x [own_x0, own_x1) it scores. Each overlap
                                        is split at its midpoint, so the owned regions of a
                                        photograph tile it exactly once: predictions stitched from
                                        them count no pixel twice.
Coordinates are in the photograph at the preset's scale. The release is MD5-checked first.
"""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a checkout without installing the package

from nwrd_public.config import PRESETS, Config  # noqa: E402
from nwrd_public.dataset import Release, open_release, read_mask  # noqa: E402
from nwrd_public.windows import (  # noqa: E402
    axis_starts,
    cut_training_windows,
    owned_spans,
    resize_mask,
    window_name,
    window_rust,
)

SPLITS = ('train', 'val', 'test')
COLUMNS = ['split', 'photo', 'file', 'y', 'x', 'size', 'scale', 'rust_px']
COLUMNS += ['own_y0', 'own_y1', 'own_x0', 'own_x1']


def index_rows(release: Release, split: str, pid: str, size: int, scale: float, kept: set):
    """The index rows of one photograph's patches that were cut (`kept`), row by row."""
    rust = resize_mask(read_mask(release.root, release.folders[pid], pid), scale)
    h, w = rust.shape
    ys, xs = axis_starts(h, size), axis_starts(w, size)
    own_y = dict(zip(ys, owned_spans(ys, size, h), strict=True))
    own_x = dict(zip(xs, owned_spans(xs, size, w), strict=True))
    positions, counts = window_rust(rust, size)
    for (y, x), rust_px in zip(positions, counts, strict=True):
        name = window_name(pid, y, x)
        if name in kept:
            yield {
                'split': split,
                'photo': pid,
                'file': f'{split}/images/{name}',
                'y': y,
                'x': x,
                'size': size,
                'scale': scale,
                'rust_px': int(rust_px),
                'own_y0': own_y[y][0],
                'own_y1': own_y[y][1],
                'own_x0': own_x[x][0],
                'own_x1': own_x[x][1],
            }


def patch(
    release: Release,
    out: str,
    preset: str = '512px',
    splits: tuple = SPLITS,
    sample: bool = False,
    workers: int = 1,
    log=print,
) -> str:
    """Cuts `splits` of the release into <out>/<split>/ and writes <out>/patches.csv; returns the
    index's path. Every patch is cut, except that with `sample` training keeps only the kernels'
    sample: every patch with rust plus a fixed 20% of the others (data seed 42). Validation and
    test are always cut whole."""
    size, scale = PRESETS[preset]['window'], PRESETS[preset]['scale']
    cfg = Config()
    rows = []
    for split in splits:
        ids = release.splits[split]
        ratio = cfg.bg_keep_ratio if sample and split == 'train' else 1.0
        dst = os.path.join(out, split)
        info = cut_training_windows(
            release, ids, dst, size, scale, ratio, cfg.data_seed, workers=workers, log=log
        )
        kept = set(info['files'])
        for pid in ids:
            rows += index_rows(release, split, pid, size, scale, kept)
    path = os.path.join(out, 'patches.csv')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    log(f'{len(rows)} patches of {size}px at {scale:g}x from {", ".join(splits)}; index {path}')
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--root', default=os.path.join(ROOT, 'data', 'raw', 'NWRD'))
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--preset', choices=sorted(PRESETS), default='512px')
    ap.add_argument('--splits', nargs='+', choices=SPLITS, default=list(SPLITS))
    ap.add_argument('--sample', action='store_true', help="training: the kernels' sample only")
    ap.add_argument('--out', default=None, help='default: build/patches/<preset>')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0, help='photographs per split, for a quick try')
    a = ap.parse_args(argv)
    release = open_release(a.root, a.manifest, a.limit, print)
    out = a.out or os.path.join(ROOT, 'build', 'patches', a.preset)
    patch(release, out, a.preset, tuple(a.splits), a.sample, a.workers)


if __name__ == '__main__':
    main()
