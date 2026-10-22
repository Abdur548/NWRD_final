"""Scoring a trained run on whole photographs.

Each validation and test photograph is predicted whole: its windows (nwrd_public/windows.py) go
through the model, and each pixel takes the probability of the window that owns it. At the
paper's scale the half-size map is resized bilinearly to the native size. Every pixel is then
scored once, at native resolution, against the full-size mask, so no padding is scored, no edge is
dropped, and both scales are scored on the same pixels.

Protocol: the binarisation threshold is chosen on the validation photographs, frozen, and applied
once to the test photographs. Test metrics at 0.5 are reported too.

Each model is also scored on exactly the NWRD paper's 2,737 test patches: 128px patches of the
half-size test photographs on a grid from the origin, in the order the authors' test.py read
them. They are scored pooled, on rust-positive patches only (the paper's Table 4 setting), and
averaged per batch of 512 as test.py did (docs/protocol.md §3).
"""

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .dataset import MEAN, STD, Release, camera, read_photo
from .metrics import ProbHistogram, per_image_counts, scores
from .models import build_model, load_checkpoint
from .windows import axis_starts, owned_spans, paper_grid, resize_mask, resize_photo, scaled_shape

PAPER_PATCH = 128  # the paper's patch size, on photographs halved in size
PAPER_BATCH = 512  # the batch size of the authors' test.py


@dataclass
class EvalPhoto:
    id: str
    image: np.ndarray  # RGB uint8 at the working scale
    mask: np.ndarray  # rust (bool) at native resolution: what is scored
    mask_work: np.ndarray  # rust (bool) at the working scale: the windows' loss targets
    camera: str


def load_photos(release: Release, ids: list[str], scale: float) -> list[EvalPhoto]:
    photos = []
    for pid in ids:
        folder = release.folders[pid]
        rgb, rust = read_photo(release.root, folder, pid)
        photos.append(
            EvalPhoto(
                pid,
                resize_photo(rgb, scale),
                rust,
                resize_mask(rust, scale),
                camera(release.root, folder, pid),
            )
        )
    return photos


def predict(
    model, photo: EvalPhoto, size: int, device, amp: bool, batch_size: int, criterion=None
) -> tuple[torch.Tensor, float, int]:
    """The photograph's rust probability at the working scale, each pixel from the window that
    owns it; the loss summed over its windows (0 without a criterion); the number of windows."""
    h, w = photo.mask_work.shape
    ys, xs = axis_starts(h, size), axis_starts(w, size)
    windows = [
        (y, x, rows, cols)
        for y, rows in zip(ys, owned_spans(ys, size, h), strict=True)
        for x, cols in zip(xs, owned_spans(xs, size, w), strict=True)
    ]
    img = torch.from_numpy(photo.image).to(device).permute(2, 0, 1)
    mean = torch.tensor(MEAN, device=device).view(3, 1, 1) * 255
    std = torch.tensor(STD, device=device).view(3, 1, 1) * 255
    target = torch.from_numpy(photo.mask_work).to(device) if criterion is not None else None
    prob = torch.empty((h, w), device=device)
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i : i + batch_size]
            inputs = torch.stack([img[:, y : y + size, x : x + size] for y, x, _, _ in batch])
            with torch.autocast(device.type, enabled=amp):
                logits = model((inputs.float() - mean) / std)
            logits = logits.float()
            if criterion is not None:
                t = torch.stack([target[y : y + size, x : x + size] for y, x, _, _ in batch])
                loss_sum += criterion(logits, t.unsqueeze(1).float()).item() * len(batch)
            p = torch.sigmoid(logits[:, 0])
            for k, (y, x, (r0, r1), (c0, c1)) in enumerate(batch):
                prob[r0:r1, c0:c1] = p[k, r0 - y : r1 - y, c0 - x : c1 - x]
    return prob, loss_sum, len(windows)


def to_native(prob: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """A working-scale probability map resized bilinearly to the native size."""
    if tuple(prob.shape) == tuple(shape):
        return prob
    return F.interpolate(prob[None, None], size=shape, mode='bilinear', align_corners=False)[0, 0]


@dataclass
class Scores:
    hist: ProbHistogram  # every native pixel of every photograph
    loss: float  # mean loss per window (nan without a criterion)
    photo_counts: dict[float, np.ndarray]  # threshold -> (photographs, 4) confusion counts
    paper_counts: dict[float, np.ndarray]  # threshold -> (paper patches, 4)
    paper_index: list[tuple[str, int, int]]  # (photograph, row, col) of each paper patch
    paper_rust: np.ndarray  # whether each paper patch has rust


def _paper_patches(prob_half: torch.Tensor, rust_half: torch.Tensor):
    """Positions and stacked (N, 1, 128, 128) probabilities and masks of the paper's patches."""
    pos = paper_grid(*prob_half.shape, PAPER_PATCH)
    s = PAPER_PATCH
    p = torch.stack([prob_half[y : y + s, x : x + s] for y, x in pos]).unsqueeze(1)
    m = torch.stack([rust_half[y : y + s, x : x + s] for y, x in pos]).unsqueeze(1)
    return pos, p, m


def score(
    model, photos: list[EvalPhoto], cfg: Config, device, criterion=None, thresholds=(), paper=False
) -> Scores:
    """Predicts each photograph whole and scores it at native resolution. Per-photograph counts
    are kept at each of `thresholds`; with `paper`, so are those of the paper's test patches, in
    the order of `photos`."""
    amp = cfg.amp and device.type == 'cuda'
    hist = ProbHistogram()
    photo_counts = {t: [] for t in thresholds}
    paper_counts = {t: [] for t in thresholds}
    paper_index, paper_rust = [], []
    loss_sum, n_windows = 0.0, 0
    for photo in photos:
        prob, loss, n = predict(
            model, photo, cfg.window, device, amp, cfg.batch_size, criterion=criterion
        )
        loss_sum, n_windows = loss_sum + loss, n_windows + n
        mask = torch.from_numpy(photo.mask).to(device)
        native = to_native(prob, photo.mask.shape)
        hist.update(native, mask)
        for t in thresholds:
            photo_counts[t].append(per_image_counts(native[None, None], mask[None, None], t)[0])
        if paper:
            # The paper's half-size photograph: the working scale itself at 0.5, otherwise the
            # native prediction resized by area, as the masks are (windows.resize_mask).
            half = scaled_shape(*photo.mask.shape, 0.5)
            if tuple(prob.shape) == half:
                prob_half, rust_half = prob, photo.mask_work
            else:
                prob_half = F.interpolate(native[None, None], size=half, mode='area')[0, 0]
                rust_half = resize_mask(photo.mask, 0.5)
            pos, p, m = _paper_patches(prob_half, torch.from_numpy(rust_half).to(device))
            paper_index += [(photo.id, y // PAPER_PATCH, x // PAPER_PATCH) for y, x in pos]
            paper_rust.append(m.flatten(1).any(1).cpu().numpy())
            for t in thresholds:
                paper_counts[t].append(per_image_counts(p, m, t))
    return Scores(
        hist=hist,
        loss=loss_sum / n_windows if criterion is not None else float('nan'),
        photo_counts={t: np.stack(v) for t, v in photo_counts.items()},
        paper_counts={t: np.concatenate(v) for t, v in paper_counts.items() if v},
        paper_index=paper_index,
        paper_rust=np.concatenate(paper_rust) if paper_rust else np.zeros(0, bool),
    )


def paper_order(ids: list[str]) -> list[str]:
    """The test photographs in the order the authors' loader read them: file names sorted as
    text (105, 123, 25, ...)."""
    return sorted(ids)


def authors_average(counts: np.ndarray, batch: int = PAPER_BATCH) -> dict[str, float]:
    """Precision, recall and F1 as the authors' test.py computed them: scikit-learn's binary
    scores on each batch of `batch` consecutive patches (a zero division counts as 0), weighted
    by the batch's number of patches."""
    total = {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    for i in range(0, len(counts), batch):
        part = counts[i : i + batch]
        tp, fp, fn, _ = (float(v) for v in part.sum(0))
        n = len(part)
        total['Precision'] += n * (tp / (tp + fp) if tp + fp else 0.0)
        total['Recall'] += n * (tp / (tp + fn) if tp + fn else 0.0)
        total['F1'] += n * (2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return {k: v / len(counts) if len(counts) else 0.0 for k, v in total.items()}


def paper_scores(counts: np.ndarray, rust: np.ndarray, batch: int = PAPER_BATCH) -> dict:
    """The paper's patches scored pooled and with the authors' averaging, over all of them and
    over those with rust only (the paper's Table 4 removed the background-only patches; the
    authors' code does not show how, so those batches are formed after the removal)."""
    return {
        'pooled': scores(*counts.sum(0)),
        'rust_only': scores(*counts[rust].sum(0)),
        'authors': authors_average(counts, batch),
        'authors_rust_only': authors_average(counts[rust], batch),
    }


def evaluate_run(cfg: Config, release: Release, run_dir: str, log=print) -> dict:
    """Picks the threshold on validation, scores test, and writes metrics.json,
    val_threshold_sweep.json, test_counts.npz and paper_patches.npz into `run_dir`."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg.model, pretrained=False, dropout=cfg.dropout).to(device)
    ckpt = load_checkpoint(model, os.path.join(run_dir, 'best.pt'), map_location=device)

    val_photos = load_photos(release, release.splits['val'], cfg.scale)
    val = score(model, val_photos, cfg, device)
    thr, val_sweep = val.hist.best_threshold(lo=cfg.thr_lo, hi=cfg.thr_hi, step=cfg.thr_step)
    at_edge = thr in (cfg.thr_lo, cfg.thr_hi)

    test_ids = paper_order(release.splits['test'])
    photos = load_photos(release, test_ids, cfg.scale)
    thresholds = tuple(dict.fromkeys((thr, 0.5)))
    test = score(model, photos, cfg, device, thresholds=thresholds, paper=True)
    pos, neg = test.hist.pos.sum(), test.hist.neg.sum()
    res = {
        'model': cfg.model,
        'window': cfg.window,
        'scale': cfg.scale,
        'best_epoch': ckpt.get('epoch'),
        'threshold': thr,
        'threshold_at_grid_edge': bool(at_edge),
        'val@thr': scores(*val.hist.counts(thr)),
        'val@0.5': scores(*val.hist.counts(0.5)),
        'test@thr': scores(*test.hist.counts(thr)),
        'test@0.5': scores(*test.hist.counts(0.5)),
        'val_photos': len(val_photos),
        'test_photos': len(photos),
        'test_rust_pixel_fraction': float(pos / (pos + neg)),
        'paper_patches': {
            'patches': len(test.paper_index),
            'rust_patches': int(test.paper_rust.sum()),
            '@thr': paper_scores(test.paper_counts[thr], test.paper_rust),
            '@0.5': paper_scores(test.paper_counts[0.5], test.paper_rust),
        },
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, 'val_threshold_sweep.json'), 'w') as f:
        json.dump(val_sweep, f)
    # Per-photograph counts drive the photograph bootstrap in compare.py; the paper patches'
    # counts let any average of them be recomputed without the model.
    np.savez_compressed(
        os.path.join(run_dir, 'test_counts.npz'),
        photos=np.array(test_ids),
        cameras=np.array([p.camera for p in photos]),
        at_thr=test.photo_counts[thr],
        at_05=test.photo_counts[0.5],
    )
    index = np.array(test.paper_index, dtype=object)
    np.savez_compressed(
        os.path.join(run_dir, 'paper_patches.npz'),
        photo=index[:, 0].astype(str),
        row=index[:, 1].astype(np.int32),
        col=index[:, 2].astype(np.int32),
        rust=test.paper_rust,
        at_thr=test.paper_counts[thr],
        at_05=test.paper_counts[0.5],
    )
    t, pp = res['test@thr'], res['paper_patches']['@0.5']['authors']
    log(
        f'[{cfg.model} {os.path.basename(run_dir)}] thr {thr:.2f}'
        f'{" (GRID EDGE)" if at_edge else ""} | test IoU {t["IoU"]:.4f} Dice {t["Dice"]:.4f} '
        f"P {t['Precision']:.4f} R {t['Recall']:.4f} | paper patches @0.5, authors' "
        f'averaging: P {pp["Precision"]:.4f} R {pp["Recall"]:.4f} F1 {pp["F1"]:.4f}'
    )
    return res
