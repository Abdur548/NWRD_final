"""The dataset study behind docs/dataset.md: sizes, cameras, masks, rust, repeated photographs,
the NWRD paper's Table 2, and the windows each scale cuts.

    python -P -E scripts/describe_data.py --root data/raw/NWRD --out results/dataset

Writes photographs.json (one record per photograph), duplicates.json (repeated photographs and
how well their masks agree) and summary.md (the tables docs/dataset.md quotes). Window counts and
the training sample use the pipeline's own functions (nwrd_public/windows.py), so they are the
numbers a run cuts. Reads every photograph once at full resolution and a few twice: a few
minutes.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a checkout without installing the package

from nwrd_public.dataset import (  # noqa: E402
    EXCLUDED_MISALIGNED,
    EXCLUDED_TWINS,
    PAPER_VAL_IDS,
    load_manifest,
    paper_validation_ids,
    photo_folders,
    splits,
    verify_release,
)
from nwrd_public.windows import (  # noqa: E402
    axis_starts,
    pos_weight,
    resize_mask,
    select_windows,
    window_name,
    window_rust,
)

Image.MAX_IMAGE_PIXELS = 60_000_000  # the photographs are 16-24 MP
SCALES = {'512px': (512, 1.0), '128px': (128, 0.5)}
# Masks placed at every offset up to PLACEMENT_MARGIN px around the photograph's corner: photo 101,
# whose mask is smaller, and as controls the release's six other cropped photographs.
PLACEMENT_CONTROLS = ('102', '103', '105', '110', '111', '149')
PLACEMENT_MARGIN = 24
# The NWRD paper's Table 2 (patches of 128px on half-size photographs; training at stride 32),
# for all 100 photographs and for its 22-photograph subset, whose photographs its README names.
TABLE2 = {'val': 2237, 'test': 2737, 'train_gp32': 327929}
TABLE2_22 = {'val': 468, 'test': 703, 'train_gp32': 72592}
SUBSET_22 = {
    'trainval': ['2', '7', '14', '30', '64', '83', '84', '90', '94', '95', '100', '118', '124',
                 '125', '132', '133', '136', '137', '138', '146'],
    'test': ['67', '123'],
}  # fmt: skip
# NWRD_final's T4 epoch times and the windows they covered, for the time estimate.
EPOCH_TIMES = os.path.join(ROOT, 'results', 'nwrd_final', 'epoch_seconds.json')
EPOCHS, SEEDS = 30, 3


def table(headers: list[str], rows: list[list]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def exif_of(img: Image.Image) -> dict:
    ex = img.getexif()
    try:
        taken = ex.get_ifd(0x8769).get(36867)  # DateTimeOriginal
    except Exception:
        taken = None

    def clean(v):
        return v.strip('\x00 ').strip() if isinstance(v, str) else v

    return {
        'make': clean(ex.get(271)),
        'camera': clean(ex.get(272)) or 'no EXIF',
        'orientation': ex.get(274),
        'taken': clean(taken),
    }


def rust_contrast(rgb: np.ndarray, rust: np.ndarray) -> float | None:
    """Mean (R - G) inside the mask minus outside: rust is orange-yellow and leaves are green, so
    an aligned mask gives a clearly positive value and a misaligned one a value near 0."""
    if rust.sum() == 0 or rust.all():
        return None
    s = rgb[..., 0].astype(np.float32) - rgb[..., 1].astype(np.float32)
    return float(s[rust].mean() - s[~rust].mean())


def placement_search(rgb: np.ndarray, rust: np.ndarray, margin: int = PLACEMENT_MARGIN) -> dict:
    """The rust contrast, at full resolution, of every placement (dy, dx) of the mask's top-left
    corner on its photograph, from `margin` px before the photograph's corner to `margin` px past
    the last placement inside it. A placement partly past an edge is scored on the part of the
    mask over the photograph. For a mask of another size, also the centred and bottom-right
    placements and the mask stretched to the photograph's size."""
    s = rgb[..., 0].astype(np.float32) - rgb[..., 1].astype(np.float32)
    big_h, big_w = s.shape
    h, w = rust.shape
    integral = np.zeros((big_h + 1, big_w + 1), np.float64)
    integral[1:, 1:] = s.cumsum(0, dtype=np.float64).cumsum(1)
    ys, xs = np.nonzero(rust)
    found = {}
    for dy in range(-margin, big_h - h + margin + 1):
        y_lo, y_hi = max(0, dy), min(big_h, dy + h)
        rows_on = (ys + dy >= 0) & (ys + dy < big_h)
        for dx in range(-margin, big_w - w + margin + 1):
            x_lo, x_hi = max(0, dx), min(big_w, dx + w)
            on = rows_on & (xs + dx >= 0) & (xs + dx < big_w)
            n_in = int(on.sum())
            inside = float(s[ys[on] + dy, xs[on] + dx].sum())
            total = (
                integral[y_hi, x_hi]
                - integral[y_lo, x_hi]
                - integral[y_hi, x_lo]
                + integral[y_lo, x_lo]
            )
            n_out = (y_hi - y_lo) * (x_hi - x_lo) - n_in
            found[(dy, dx)] = inside / n_in - (total - inside) / n_out
    best = max(found, key=found.get)
    out = {
        'photo_wh': [big_w, big_h],
        'mask_wh': [w, h],
        'margin': margin,
        'best_yx': list(best),
        'best': round(found[best], 2),
        'top_left': round(found[(0, 0)], 2),
    }
    if (h, w) != (big_h, big_w):
        stretched = Image.fromarray(rust.astype(np.uint8) * 255).resize(
            (big_w, big_h), Image.NEAREST
        )
        out |= {
            'centre': round(found[((big_h - h) // 2, (big_w - w) // 2)], 2),
            'bottom_right': round(found[(big_h - h, big_w - w)], 2),
            'stretched': round(rust_contrast(rgb, np.asarray(stretched) > 127), 2),
        }
    return out


def placement_figure(
    rgb: np.ndarray, rust: np.ndarray, best_yx: list[int], path: str, crops=3, half=70, zoom=3
) -> None:
    """Close-ups of the mask's largest annotations, one row each: the photograph, the mask's
    outline with its corner at the photograph's (cyan) and at the best placement (magenta)."""
    h, w = rust.shape
    big_h, big_w = rgb.shape[:2]

    def placed(dy: int, dx: int) -> np.ndarray:
        out = np.zeros((big_h, big_w), np.uint8)
        ys = slice(max(0, dy), min(big_h, h + dy))
        xs = slice(max(0, dx), min(big_w, w + dx))
        out[ys, xs] = rust[ys.start - dy : ys.stop - dy, xs.start - dx : xs.stop - dx]
        return out

    at_corner, at_best = placed(0, 0), placed(*best_yx)
    _, _, stats, centres = cv2.connectedComponentsWithStats(at_corner)
    rows = []
    for label in np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:crops] + 1:
        cx, cy = (int(v) for v in centres[label])
        y0 = min(max(0, cy - half), big_h - 2 * half)
        x0 = min(max(0, cx - half), big_w - 2 * half)
        window = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        crop = cv2.resize(rgb[window], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        panels = [crop]
        for mask, colour in ((at_corner, (0, 255, 255)), (at_best, (255, 0, 255))):
            big = cv2.resize(mask[window], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            panels.append(cv2.drawContours(crop.copy(), contours, -1, colour, 1))
        gap = np.full((crop.shape[0], 6, 3), 255, np.uint8)
        rows.append(np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1))
    gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([part for row in rows for part in (row, gap)][:-1])).save(path)


def windows_of(rust: np.ndarray) -> tuple[dict, dict]:
    """Per scale: the fitted grid, rust windows, overlap and the padding a padded grid would add,
    and every window's rust count (for the training sample)."""
    out, stats = {}, {}
    for name, (size, scale) in SCALES.items():
        mask = resize_mask(rust, scale)
        h, w = mask.shape
        positions, counts = window_rust(mask, size)
        ys, xs = axis_starts(h, size), axis_starts(w, size)
        out[name] = {
            'grid': [len(ys), len(xs)],
            'windows': len(positions),
            'rust_windows': int((counts > 0).sum()),
            'overlap': [
                max((a + size - b for a, b in itertools.pairwise(s)), default=0) for s in (ys, xs)
            ],
            'padding': [math.ceil(n / size) * size - n for n in (h, w)],
        }
        stats[name] = {(y, x): int(c) for (y, x), c in zip(positions, counts, strict=True)}
    return out, stats


def paper_patches(rust: np.ndarray) -> dict:
    """The NWRD paper's patches of the half-size photograph: 128px from the origin, the remainder
    dropped (validation and test), and the stride-32 grid of its training (GP-32)."""
    half = resize_mask(rust, 0.5)
    h2, w2 = half.shape
    rows, cols = h2 // 128, w2 // 128
    tiles = half[: rows * 128, : cols * 128].reshape(rows, 128, cols, 128)
    return {
        'half_wh': [w2, h2],
        'patches': rows * cols,
        'rust_patches': int(tiles.any(axis=(1, 3)).sum()),
        'dropped_px': [w2 - cols * 128, h2 - rows * 128],
        'gp32': ((h2 - 128) // 32 + 1) * ((w2 - 128) // 32 + 1),
    }


def describe(root: str, manifest: dict) -> tuple[list[dict], dict, dict]:
    """One record per photograph, plus each photograph's 1/8 grey thumbnail and per-window rust
    counts, kept in memory for the repeat search and the training sample."""
    folders = photo_folders(manifest)
    ours = splits(manifest)
    our_split = {p: s for s, ids in ours.items() for p in ids}
    train_images = [n for n in manifest['files'] if n.startswith('train/images/')]
    paper_val = set(paper_validation_ids(train_images))
    records, thumbs, window_stats = [], {}, {}
    for pid in sorted(folders, key=int):
        folder = folders[pid]
        img_path = os.path.join(root, folder, 'images', f'{pid}.jpg')
        with Image.open(img_path) as img:
            rec = {'id': pid, 'folder': folder, **exif_of(img)}
            rgb = np.asarray(img.convert('RGB'))
        with Image.open(os.path.join(root, folder, 'masks', f'{pid}.png')) as m:
            mask_mode, grey = m.mode, np.asarray(m.convert('L'))
        rust = grey > 127
        rec.update(
            {
                'paper_split': 'val' if pid in paper_val else folder,
                'split': our_split.get(pid)
                or (
                    'excluded (mask misaligned)'
                    if pid in EXCLUDED_MISALIGNED
                    else f'excluded (twin of {EXCLUDED_TWINS[pid]})'
                ),
                'bytes': os.path.getsize(img_path),
                'pixel_md5': hashlib.md5(rgb.tobytes()).hexdigest(),
                'photo_wh': [rgb.shape[1], rgb.shape[0]],
                'mask_wh': [rust.shape[1], rust.shape[0]],
                'mask_mode': mask_mode,
                'mask_values': sorted(int(v) for v in np.unique(grey)),
                'rust_px': int(rust.sum()),
                'pixels': int(rust.size),
                'rust_share': float(rust.mean()),
            }
        )
        if rust.shape == rgb.shape[:2]:
            small = np.asarray(Image.fromarray(rgb).reduce(8))
            small_rust = np.asarray(Image.fromarray(rust.astype(np.uint8) * 255).reduce(8)) > 127
            rec['contrast'] = rust_contrast(small, small_rust)
        if rust.shape != rgb.shape[:2] or pid in PLACEMENT_CONTROLS:
            rec['placement'] = placement_search(rgb, rust)
        rec['windows'], window_stats[pid] = windows_of(rust)
        rec['paper'] = paper_patches(rust)
        records.append(rec)
        grey_rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        thumbs[pid] = cv2.resize(
            grey_rgb, (rgb.shape[1] // 8, rgb.shape[0] // 8), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        del rgb, grey_rgb
        print(
            f'{folder:5s} {pid:>4s} {rec["camera"]:>11s} {rec["photo_wh"]} mask {rec["mask_wh"]} '
            f'rust {100 * rec["rust_share"]:.2f}% {rec["split"]}',
            flush=True,
        )
    return records, thumbs, window_stats


def _rgb(root: str, folder: str, pid: str) -> np.ndarray:
    with Image.open(os.path.join(root, folder, 'images', f'{pid}.jpg')) as img:
        return np.asarray(img.convert('RGB')).astype(np.int16)


def _mask(root: str, folder: str, pid: str) -> np.ndarray:
    with Image.open(os.path.join(root, folder, 'masks', f'{pid}.png')) as m:
        return np.asarray(m.convert('L')) > 127


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def find_repeats(root: str, records: list[dict], thumbs: dict) -> dict:
    """Photographs whose decoded pixels are identical, near-identical ones (same size,
    thumbnails correlated above 0.98) and crops of another photograph; for each, how well the
    two masks agree."""
    by_id = {r['id']: r for r in records}
    out = {'identical': [], 'near': [], 'crops': []}
    groups = {}
    for r in records:
        groups.setdefault(r['pixel_md5'], []).append(r['id'])
    for ids in (g for g in groups.values() if len(g) > 1):
        a, b = (by_id[i] for i in ids[:2])
        with open(os.path.join(root, a['folder'], 'images', f'{a["id"]}.jpg'), 'rb') as f:
            ra = f.read()
        with open(os.path.join(root, b['folder'], 'images', f'{b["id"]}.jpg'), 'rb') as f:
            rb = f.read()
        differing = sum(x != y for x, y in zip(ra, rb, strict=False)) + abs(len(ra) - len(rb))
        agreement = _iou(_mask(root, a['folder'], a['id']), _mask(root, b['folder'], b['id']))
        out['identical'].append(
            {
                'ids': ids,
                'paper_splits': [by_id[i]['paper_split'] for i in ids],
                'bytes_differing': differing,
                'camera': a['camera'],
                'taken': [a['taken'], b['taken']],
                'mask_iou': round(agreement, 3),
                'rust_px': [a['rust_px'], b['rust_px']],
            }
        )
    ids = sorted(thumbs, key=int)
    for a, b in itertools.combinations(ids, 2):
        ta, tb = thumbs[a], thumbs[b]
        if ta.shape != tb.shape or by_id[a]['pixel_md5'] == by_id[b]['pixel_md5']:
            continue
        r = float(np.corrcoef(ta.ravel(), tb.ravel())[0, 1])
        if r > 0.98:
            out['near'].append({'ids': [a, b], 'r': round(r, 4)})
    for a, b in itertools.permutations(ids, 2):
        small, big = thumbs[a], thumbs[b]
        if (
            small.shape == big.shape
            or small.shape[0] > big.shape[0]
            or small.shape[1] > big.shape[1]
        ):
            continue
        _, found, _, (x8, y8) = cv2.minMaxLoc(cv2.matchTemplate(big, small, cv2.TM_CCOEFF_NORMED))
        if found < 0.95:
            continue
        # The exact offset near the thumbnails' match, on every 8th pixel, then the difference
        # over all pixels at that offset.
        ra, rb = _rgb(root, by_id[a]['folder'], a), _rgb(root, by_id[b]['folder'], b)
        h, w = ra.shape[:2]
        _, dy, dx = min(
            (float(np.abs(ra[::8, ::8] - rb[dy : dy + h : 8, dx : dx + w : 8]).mean()), dy, dx)
            for dy in range(max(0, 8 * y8 - 8), min(rb.shape[0] - h, 8 * y8 + 8) + 1)
            for dx in range(max(0, 8 * x8 - 8), min(rb.shape[1] - w, 8 * x8 + 8) + 1)
        )
        diff = np.abs(ra - rb[dy : dy + h, dx : dx + w])
        mask_a = _mask(root, by_id[a]['folder'], a)
        mask_b = _mask(root, by_id[b]['folder'], b)[dy : dy + h, dx : dx + w]
        out['crops'].append(
            {
                'crop': a,
                'of': b,
                'paper_splits': [by_id[a]['paper_split'], by_id[b]['paper_split']],
                'xy': [dx, dy],
                'mean_abs_difference': round(float(diff.mean()), 2),  # per colour channel
                'identical_pixels': round(float((diff.max(2) == 0).mean()), 3),
                'max_difference': int(diff.max()),
                'mask_iou': round(_iou(mask_a, mask_b), 3),
            }
        )
    return out


def training_sample(records: list[dict], window_stats: dict) -> dict:
    """The training sample each scale cuts (windows.select_windows, data seed 42, 20% of the
    background windows) and the BCE rust weight it gives."""
    out = {}
    train = [r['id'] for r in records if r['split'] == 'train']
    for name, (size, _) in SCALES.items():
        stats = {
            window_name(p, y, x): {'pos': c, 'total': size * size}
            for p in train
            for (y, x), c in window_stats[p][name].items()
        }
        files, sel = select_windows(stats, 0.20, 42)
        pos = sum(stats[f]['pos'] for f in files)
        out[name] = {
            'photos': len(train),
            'windows': len(stats),
            **sel,
            'sample': len(files),
            'rust_pixel_share': pos / (len(files) * size * size),
            'pos_weight': pos_weight(stats, files),
            'val_windows': sum(
                r['windows'][name]['windows'] for r in records if r['split'] == 'val'
            ),
        }
    return out


def gpu_hours(sample: dict, timing: dict) -> dict:
    """T4 hours per model and scale for 3 seeds of 30 epochs: NWRD_final's mean epoch time,
    scaled by the windows an epoch processes; a validation window counts as a third of a
    training window (a forward pass only)."""
    out = {}
    for scale, models in timing['scales'].items():
        ours = sample[scale]['sample'] + sample[scale]['val_windows'] / 3
        out[scale] = {
            m: t['epoch_seconds']
            * ours
            / (t['train_windows'] + t['val_windows'] / 3)
            * EPOCHS
            * SEEDS
            / 3600
            for m, t in models.items()
        }
    return out


def _pct(x: float) -> str:
    return f'{100 * x:.2f}%'


def write_summary(
    records: list[dict], repeats: dict, sample: dict, manifest: dict, timing: dict
) -> str:
    by_id = {r['id']: r for r in records}
    L = [
        '# The public NWRD release: dataset study',
        '',
        f'Generated by `scripts/describe_data.py` from the release checked against '
        f'`data/manifest.json` ({len(manifest["files"])} files; {manifest["source"]}).',
        '',
        "## Sizes and cameras (split as the NWRD paper's code splits it)",
        '',
    ]
    classes = Counter()
    for r in records:
        classes[(tuple(r['photo_wh']), r['camera'], r['paper_split'])] += 1
    rows = []
    for size in sorted(
        {k[0] for k in classes}, key=lambda s: -sum(v for k, v in classes.items() if k[0] == s)
    ):
        cams = sorted({k[1] for k in classes if k[0] == size})
        for cam in cams:
            n = {s: classes[(size, cam, s)] for s in ('train', 'val', 'test')}
            ids = [r['id'] for r in records if tuple(r['photo_wh']) == size and r['camera'] == cam]
            one = by_id[ids[0]]
            rows.append([
                f'{size[0]}×{size[1]}', cam, n['train'], n['val'], n['test'], sum(n.values()),
                one['windows']['512px']['windows'], one['windows']['128px']['windows'],
                ', '.join(ids) if len(ids) == 1 else '',
            ])  # fmt: skip
    L.append(table(['Photo W×H', 'Camera', 'Train', 'Val', 'Test', 'All', '512px windows',
                    '128px windows at 0.5×', 'Photograph'], rows))  # fmt: skip
    orient = Counter(str(r['orientation']) for r in records)
    values = Counter(tuple(r['mask_values']) for r in records)
    contrasts = [r['contrast'] for r in records if r.get('contrast') is not None]
    L += [
        '',
        '## EXIF, masks and alignment',
        '',
        f'- EXIF orientation: {dict(orient)} (None = no EXIF).',
        f'- Mask modes: {dict(Counter(r["mask_mode"] for r in records))}; mask values: '
        f'{ {str(list(k)): v for k, v in values.items()} }.',
        f"- Rust contrast of the {len(contrasts)} masks with their photograph's size (mean R − G "
        f'inside minus outside, at 1/8 scale): min {min(contrasts):.1f}, median '
        f'{statistics.median(contrasts):.1f}, max {max(contrasts):.1f}.',
    ]
    placed = [r for r in records if 'placement' in r]
    for r in placed:
        p = r['placement']
        if 'stretched' in p:
            L.append(
                f'- Photo {r["id"]}: mask {p["mask_wh"][0]}×{p["mask_wh"][1]} on a '
                f'{p["photo_wh"][0]}×{p["photo_wh"][1]} photograph. Contrast at the top left '
                f'{p["top_left"]}, centred {p["centre"]}, bottom right {p["bottom_right"]}, '
                f'stretched {p["stretched"]}; best placement (y, x) = {tuple(p["best_yx"])} at '
                f'{p["best"]}. Split: {r["split"]}.'
            )
    L += [
        '',
        f'Every placement of the mask\'s top-left corner (y, x) up to {PLACEMENT_MARGIN} px around '
        "the photograph's (partly past an edge: scored on the part over the photograph), rust "
        'contrast at full resolution:',
        '',
        table(['Photograph', 'Photograph W×H', 'Mask W×H', 'Best placement (y, x)',
               'Contrast there', 'At (0, 0)'],
              [[r['id'], '×'.join(map(str, r['placement']['photo_wh'])),
                '×'.join(map(str, r['placement']['mask_wh'])),
                str(tuple(r['placement']['best_yx'])), r['placement']['best'],
                r['placement']['top_left']] for r in placed]),
    ]  # fmt: skip
    L += ['', '## Rust', '']
    rows = []
    for label, sel in (
        ("Train (the paper's)", [r for r in records if r['paper_split'] == 'train']),
        ('Val', [r for r in records if r['paper_split'] == 'val']),
        ('Test', [r for r in records if r['paper_split'] == 'test']),
        ('All', records),
        ('Train (ours: without the four twins)', [r for r in records if r['split'] == 'train']),
    ):
        f = [r['rust_share'] for r in sel]
        pooled = sum(r['rust_px'] for r in sel) / sum(r['pixels'] for r in sel)
        rows.append([label, len(sel), sum(x == 0 for x in f), _pct(min(f)),
                     _pct(statistics.median(f)), _pct(max(f)), _pct(pooled)])  # fmt: skip
    L.append(
        table(['Split', 'Photographs', 'Without rust', 'Min', 'Median', 'Max', 'Pooled'], rows)
    )
    by_camera = {}
    for r in records:
        by_camera.setdefault(r['camera'], []).append(r['rust_share'])
    medians = [f'{c} ({len(v)}) {_pct(statistics.median(v))}' for c, v in by_camera.items()]
    L += ['', 'By camera (median rust share): ' + '; '.join(medians) + '.']

    ious = [p['mask_iou'] for p in repeats['identical']] + [c['mask_iou'] for c in repeats['crops']]
    scenes = len(records) - len(repeats['identical']) - len(repeats['crops'])
    L += [
        '',
        '## Repeated photographs',
        '',
        f'{len(repeats["identical"])} pairs decode to identical pixels; '
        f'{len(repeats["crops"])} photograph is a crop of another; {len(repeats["near"])} other '
        f'pairs are near-identical. So the release holds {scenes} distinct scenes. Each copy has '
        'its own mask.',
        '',
        table(['Photographs', 'Paper\'s splits', 'Bytes differing', 'Mask IoU', 'Rust pixels'], [
            [' / '.join(p['ids']), ' / '.join(p['paper_splits']), p['bytes_differing'],
             p['mask_iou'], ' / '.join(map(str, p['rust_px']))] for p in repeats['identical']
        ] + [
            [f'{c["crop"]} ⊂ {c["of"]} at (x {c["xy"][0]}, y {c["xy"][1]}), saved again: mean '
             f'difference {c["mean_abs_difference"]} per colour channel (max '
             f'{c["max_difference"]}), {100 * c["identical_pixels"]:.0f}% of pixels identical',
             ' / '.join(c['paper_splits']), '–',
             c['mask_iou'], '–'] for c in repeats['crops']
        ]),
        '',
        f'Twin masks agree at IoU {min(ious):.2f} to {max(ious):.2f} (median '
        f'{statistics.median(ious):.2f}, {len(ious)} pairs).',
    ]  # fmt: skip
    test = [r for r in records if r['paper_split'] == 'test']
    val = [r for r in records if r['paper_split'] == 'val']
    leaked_test = [by_id[t] for t in EXCLUDED_TWINS.values() if by_id[t]['paper_split'] == 'test']
    leaked_val = [by_id[t] for t in EXCLUDED_TWINS.values() if by_id[t]['paper_split'] == 'val']

    def share(sel, of, key):
        return _pct(sum(r[key] for r in sel) / sum(r[key] for r in of))

    L += [
        '',
        f"Under the paper's split, test photographs {', '.join(r['id'] for r in leaked_test)} have "
        f'their twin in training: they hold {share(leaked_test, test, "pixels")} of the test '
        f'pixels and {share(leaked_test, test, "rust_px")} of the test rust. Validation '
        f'photographs {", ".join(r["id"] for r in leaked_val)} hold '
        f'{share(leaked_val, val, "rust_px")} of the validation rust. Our training set leaves out '
        f'their twins {", ".join(sorted(EXCLUDED_TWINS, key=int))}.',
    ]

    def table2(train: list[str], val: list[str], test: list[str]) -> dict:
        def n(ids, key):
            return sum(by_id[i]['paper'][key] for i in ids)

        return {
            'val': n(val, 'patches'),
            'test': n(test, 'patches'),
            'train_gp32': n(train, 'gp32'),
        }

    paper_ids = {
        s: [r['id'] for r in records if r['paper_split'] == s] for s in ('train', 'val', 'test')
    }
    got = table2(paper_ids['train'], paper_ids['val'], paper_ids['test'])
    val22 = paper_validation_ids([f'train/images/{i}.jpg' for i in SUBSET_22['trainval']])
    tr22 = [i for i in SUBSET_22['trainval'] if i not in val22]
    got22 = table2(tr22, val22, SUBSET_22['test'])
    L += [
        '',
        '## The NWRD paper\'s Table 2, recomputed',
        '',
        'Its validation photographs (seed-0 draw): '
        f'{", ".join(sorted(PAPER_VAL_IDS, key=int))}; test: '
        "the release's test/ folder. 128px patches of the half-size photographs, from the "
        'origin, the remainder dropped; training at stride 32 (GP-32).',
        '',
        table(['', 'Paper, 100 photographs', 'Recomputed', 'Paper, 22 photographs', 'Recomputed'],
              [[k, f'{TABLE2[k]:,}', f'{got[k]:,}', f'{TABLE2_22[k]:,}', f'{got22[k]:,}']
               for k in TABLE2]),
        '',
        f'The 22-photograph subset\'s validation draw: {", ".join(sorted(val22, key=int))}.',
    ]  # fmt: skip

    L += ['', '## Windows per size', '']
    rows, seen = [], set()
    for r in sorted(records, key=lambda r: tuple(r['mask_wh'])):
        key = tuple(r['mask_wh'])
        if key in seen:
            continue
        seen.add(key)
        a, b = r['windows']['512px'], r['windows']['128px']
        rows.append([
            f'{key[0]}×{key[1]}', sum(tuple(x['mask_wh']) == key for x in records),
            f'{a["grid"][1]}×{a["grid"][0]} = {a["windows"]}', f'{max(a["overlap"])} px',
            f'{max(a["padding"])} px', f'{b["grid"][1]}×{b["grid"][0]} = {b["windows"]}',
            f'{max(b["overlap"])} px', r['paper']['patches'],
        ])  # fmt: skip
    L.append(table(['Scored size (mask) W×H', 'Photographs', '512px: windows (cols×rows)',
                    'Largest overlap', 'Padding a padded grid adds', '128px at 0.5×: windows',
                    'Largest overlap', 'Paper patches'], rows))  # fmt: skip

    L += ['', '## Our split (D3): windows per split', '']
    rows = []
    for s in ('train', 'val', 'test'):
        sel = [r for r in records if r['split'] == s]
        counts = [sum(r['windows'][k][f] for r in sel) for k in SCALES
                  for f in ('windows', 'rust_windows')]  # fmt: skip
        paper = [sum(r['paper'][f] for r in sel) for f in ('patches', 'rust_patches')]
        rows.append([s, len(sel), *counts, *paper])
    L.append(table(['Split', 'Photographs', '512px windows', 'with rust', '128px windows',
                    'with rust', 'Paper patches', 'with rust'], rows))  # fmt: skip

    L += ['', '## Training sample (every rust window + 20% of the others, data seed 42)', '']
    L.append(table(
        ['Scale', 'Photographs', 'Windows', 'With rust', 'Background kept', 'Sample per epoch',
         'Rust share of its pixels', 'BCE pos_weight', 'NWRD_final\'s sample'],
        [[k, v['photos'], f'{v["windows"]:,}', f'{v["rust_windows"]:,}',
          f'{v["bg_kept"]:,} of {v["bg_total"]:,}', f'{v["sample"]:,}', _pct(v['rust_pixel_share']),
          f'{v["pos_weight"]:.1f}', f'{timing["scales"][k]["canet_b4"]["train_windows"]:,}']
         for k, v in sample.items()],
    ))  # fmt: skip
    hours = gpu_hours(sample, timing)
    models = list(next(iter(hours.values())))
    L += [
        '',
        f'## T4 time estimate ({SEEDS} seeds of {EPOCHS} epochs)',
        '',
        "From `NWRD_final`'s mean epoch times on a Tesla T4 "
        '(`results/nwrd_final/epoch_seconds.json`), scaled by the windows an epoch '
        'processes here; a validation window counts as a third of a training window (a forward '
        'pass only).',
        '',
        table(['Scale', *models, 'All three'],
              [[s, *(f'{h[m]:.1f} h' for m in models), f'{sum(h.values()):.1f} h']
               for s, h in hours.items()]),
        '',
        f'In all: {sum(sum(h.values()) for h in hours.values()):.1f} GPU-hours.',
    ]  # fmt: skip
    return '\n'.join(L) + '\n'


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--root', default=os.path.join(ROOT, 'data', 'raw', 'NWRD'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'dataset'))
    ap.add_argument('--manifest', default=None)
    a = ap.parse_args(argv)
    manifest = load_manifest(a.manifest)
    verify_release(a.root, manifest)  # the report says the release was checked
    records, thumbs, window_stats = describe(a.root, manifest)
    repeats = find_repeats(a.root, records, thumbs)
    sample = training_sample(records, window_stats)
    os.makedirs(a.out, exist_ok=True)
    for name, data in (('photographs.json', records), ('duplicates.json', repeats)):
        with open(os.path.join(a.out, name), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(data, f, indent=1)
            f.write('\n')
    for r in (r for r in records if 'stretched' in r.get('placement', {})):  # photo 101
        with Image.open(os.path.join(a.root, r['folder'], 'images', f'{r["id"]}.jpg')) as img:
            rgb = np.asarray(img.convert('RGB'))
        with Image.open(os.path.join(a.root, r['folder'], 'masks', f'{r["id"]}.png')) as m:
            rust = np.asarray(m.convert('L')) > 127
        figure = os.path.join(a.out, f'mask_{r["id"]}.png')
        placement_figure(rgb, rust, r['placement']['best_yx'], figure)
    with open(EPOCH_TIMES, encoding='utf-8') as f:
        timing = json.load(f)
    summary = write_summary(records, repeats, sample, manifest, timing)
    with open(os.path.join(a.out, 'summary.md'), 'w', encoding='utf-8', newline='\n') as f:
        f.write(summary)
    print(summary)


if __name__ == '__main__':
    main()
