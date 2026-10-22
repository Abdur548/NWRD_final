"""Windows of a fixed size on a stride fitted to each photograph, and the training windows' cut.

The release's photographs have 10 sizes, and none has both sides a whole number of 512 px. On a
side of L px, ceil(L / P) windows of P px are spread evenly from 0 to L - P. No padding is
needed, every pixel is covered, and neighbours overlap only by what L lacks of a whole number of
windows (on the photographs used, at most 46 px at 512px and 11 px at 128px). For scoring, each
overlap is split at its midpoint, so every pixel is scored exactly once, by the window whose
centre is nearer (docs/protocol.md §3).

At the NWRD paper's scale a photograph is first resized by 0.5 with an anti-aliased Lanczos filter,
and its mask by area (a half-size pixel is rust when at least half of the area it replaces is).
"""

import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise

import cv2
import numpy as np
from PIL import Image

from .dataset import Release, read_mask, read_photo

# How photographs are resized; recorded with every cut, so a cut made another way is not reused.
PHOTO_RESIZE = 'Pillow Lanczos'


def axis_starts(length: int, size: int) -> list[int]:
    """Start positions of the windows along one side of `length` px."""
    if length < size:
        raise ValueError(
            f'a side of {length} px is shorter than a {size} px window; windows are never padded'
        )
    n = math.ceil(length / size)
    if n == 1:
        return [0]
    return [round(i * (length - size) / (n - 1)) for i in range(n)]


def grid(height: int, width: int, size: int) -> list[tuple[int, int]]:
    """Top-left corners (y, x) of a photograph's windows, row by row."""
    return [(y, x) for y in axis_starts(height, size) for x in axis_starts(width, size)]


def owned_spans(starts: list[int], size: int, length: int) -> list[tuple[int, int]]:
    """For each window along one side, the pixels [lo, hi) it scores: each overlap is split at its
    midpoint, so every pixel belongs to exactly one window, the one whose centre is nearer."""
    bounds = [0] + [(a + b + size) // 2 for a, b in pairwise(starts)] + [length]
    return list(pairwise(bounds))


def scaled_shape(height: int, width: int, scale: float) -> tuple[int, int]:
    return round(height * scale), round(width * scale)


def resize_photo(rgb: np.ndarray, scale: float) -> np.ndarray:
    """The photograph resized by `scale` with a Lanczos filter widened by the reduction, so it
    removes what half the resolution cannot hold: the anti-aliasing the NWRD paper describes.
    (OpenCV's INTER_LANCZOS4 keeps a fixed 8-tap kernel when shrinking and lets fine texture fold
    back into false patterns.)"""
    if scale == 1:
        return rgb
    h, w = scaled_shape(*rgb.shape[:2], scale)
    return np.array(Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS))


def resize_mask(rust: np.ndarray, scale: float) -> np.ndarray:
    """The mask resized by `scale`: a pixel is rust when at least half of its area was rust."""
    if scale == 1:
        return rust
    h, w = scaled_shape(*rust.shape, scale)
    return cv2.resize(rust.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= 0.5


def window_rust(rust: np.ndarray, size: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    """The windows of a mask and the number of rust pixels in each, from its integral image."""
    h, w = rust.shape
    integral = np.zeros((h + 1, w + 1), np.int64)
    np.cumsum(np.cumsum(rust, 0, dtype=np.int64), 1, out=integral[1:, 1:])
    positions = grid(h, w, size)
    ys = np.array([y for y, _ in positions])
    xs = np.array([x for _, x in positions])
    counts = (
        integral[ys + size, xs + size]
        - integral[ys, xs + size]
        - integral[ys + size, xs]
        + integral[ys, xs]
    )
    return positions, counts


def paper_grid(height: int, width: int, size: int = 128) -> list[tuple[int, int]]:
    """The NWRD paper's test patches of a (half-size) photograph, row by row: a grid from the
    origin with stride `size`; what does not fill a whole patch at the right and bottom is not
    scored."""
    return [(r * size, c * size) for r in range(height // size) for c in range(width // size)]


def window_name(pid: str, y: int, x: int) -> str:
    return f'{pid}_y{y:04d}_x{x:04d}.png'


def select_windows(stats: dict, bg_keep_ratio: float, seed: int) -> tuple[list[str], dict]:
    """All windows with rust plus a seeded random fraction of the background-only ones.

    The seed is the *data* seed, fixed across models so every model sees the same windows."""
    rust = sorted(f for f, s in stats.items() if s['pos'] > 0)
    background = sorted(f for f, s in stats.items() if s['pos'] == 0)
    kept = random.Random(seed).sample(background, int(len(background) * bg_keep_ratio))
    info = {'rust_windows': len(rust), 'bg_kept': len(kept), 'bg_total': len(background)}
    return sorted(rust + kept), info


def pos_weight(stats: dict, files: list[str]) -> float:
    """Background-to-rust pixel ratio over `files`: the BCE positive-class weight."""
    pos = sum(stats[f]['pos'] for f in files)
    total = sum(stats[f]['total'] for f in files)
    return (total - pos) / max(pos, 1)


def _count_rust(job) -> list[tuple[str, str, int, int, int, int]]:
    """(photograph, name, y, x, rust pixels, pixels) of every window of one photograph, counted
    on its mask."""
    root, folder, pid, size, scale = job
    positions, counts = window_rust(resize_mask(read_mask(root, folder, pid), scale), size)
    return [
        (pid, window_name(pid, y, x), y, x, int(c), size * size)
        for (y, x), c in zip(positions, counts, strict=True)
    ]


def _cut_photo(job) -> int:
    """Writes the selected windows of one photograph as lossless PNG crops."""
    root, folder, pid, size, scale, windows, dst = job
    rgb, rust = read_photo(root, folder, pid)
    rgb, rust = resize_photo(rgb, scale), resize_mask(rust, scale)
    for name, y, x in windows:
        for path, arr in (
            (
                os.path.join(dst, 'images', name),
                cv2.cvtColor(rgb[y : y + size, x : x + size], cv2.COLOR_RGB2BGR),
            ),
            (
                os.path.join(dst, 'masks', name),
                rust[y : y + size, x : x + size].astype(np.uint8) * 255,
            ),
        ):
            # OpenCV reports a failed write (a full disk, a path it cannot encode) by returning
            # False, not by raising; a cut with a missing window must not be cached as complete.
            if not cv2.imwrite(path, arr):
                raise OSError(f'could not write {path}')
    return len(windows)


def _map(fn, jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) < 2:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        return list(ex.map(fn, jobs))


def cut_training_windows(
    release: Release,
    ids: list[str],
    dst: str,
    size: int,
    scale: float,
    bg_keep_ratio: float,
    seed: int,
    workers: int = 1,
    log=print,
) -> dict:
    """Cuts the training sample of photographs `ids` into <dst>/{images,masks}/ and returns its
    description, also written to <dst>/windows.json once the cut is complete, with the key
    'dir' (= dst) added.

    Every window of every photograph is counted from the masks; the sample is every window with
    rust plus a `bg_keep_ratio` share of the others (`select_windows`); only the sample is cut.
    Cutting is deterministic, so a finished cut with the same settings is reused, and one with
    other settings is refused rather than overwritten."""
    params = {
        'source': os.path.abspath(release.root),
        'photos': list(ids),
        'window': size,
        'scale': scale,
        'photo_resize': PHOTO_RESIZE,
        'bg_keep_ratio': bg_keep_ratio,
        'seed': seed,
    }
    marker = os.path.join(dst, 'windows.json')
    if os.path.exists(marker):
        with open(marker, encoding='utf-8') as f:
            done = json.load(f)
        if done['params'] != params:
            raise RuntimeError(f'{dst} holds windows cut with other settings; use another dir')
        log(f'windows: reusing {dst}')
        return {**done, 'dir': dst}

    t0 = time.time()
    jobs = [(release.root, release.folders[p], p, size, scale) for p in ids]
    rows = [r for photo in _map(_count_rust, jobs, workers) for r in photo]
    stats = {name: {'pos': pos, 'total': total} for _, name, _, _, pos, total in rows}
    files, sel = select_windows(stats, bg_keep_ratio, seed)
    chosen = set(files)
    by_photo = {}
    for pid, name, y, x, _, _ in rows:
        if name in chosen:
            by_photo.setdefault(pid, []).append((name, y, x))
    for kind in ('images', 'masks'):
        os.makedirs(os.path.join(dst, kind), exist_ok=True)
    cut = [(release.root, release.folders[p], p, size, scale, w, dst) for p, w in by_photo.items()]
    _map(_cut_photo, cut, workers)

    pos = sum(stats[f]['pos'] for f in files)
    info = {
        'params': params,
        'photos': len(ids),
        'windows': len(stats),
        **sel,
        'sample': len(files),
        'rust_pixel_fraction': pos / sum(stats[f]['total'] for f in files),
        'pos_weight': pos_weight(stats, files),
        'seconds': round(time.time() - t0, 1),
        'files': files,
    }
    # Written last and atomically: its presence means every window above was written.
    with open(marker + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=1)
    os.replace(marker + '.tmp', marker)
    log(
        f'windows: {len(ids)} photographs -> {len(stats)} windows of {size}px at {scale:g}x, '
        f'{sel["rust_windows"]} with rust; sample {len(files)} (pos_weight '
        f'{info["pos_weight"]:.1f}), cut in {info["seconds"]:.0f}s to {dst}'
    )
    return {**info, 'dir': dst}
