"""Whole-photograph scoring: each pixel predicted by the window that owns it, scored once at native
resolution; the NWRD paper's test patches and the authors' per-batch averaging."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from nwrd_public.config import Config
from nwrd_public.evaluate import (
    EvalPhoto,
    authors_average,
    paper_order,
    paper_scores,
    predict,
    score,
    to_native,
)
from nwrd_public.windows import axis_starts, owned_spans, resize_mask, resize_photo

CPU = torch.device('cpu')


class ColumnInWindow(nn.Module):
    """Logit = column within the window / window width: the assembled map shows which window
    scored each pixel."""

    def forward(self, x):
        b, _, h, w = x.shape
        return (torch.arange(w, dtype=torch.float32) / w).expand(b, 1, h, w).clone()


class Constant(nn.Module):
    def __init__(self, logit: float):
        super().__init__()
        self.logit = logit

    def forward(self, x):
        return torch.full((x.shape[0], 1, *x.shape[2:]), self.logit)


def _photo(h: int, w: int, scale: float, seed: int = 0) -> EvalPhoto:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    rust = rng.random((h, w)) > 0.9
    return EvalPhoto(str(seed), resize_photo(rgb, scale), rust, resize_mask(rust, scale), 'cam')


def test_assembly_takes_each_pixel_from_its_owner():
    photo = _photo(300, 700, 1.0)
    prob, _, n = predict(ColumnInWindow(), photo, 128, CPU, False, batch_size=3)
    starts = axis_starts(700, 128)
    expected = np.zeros(700, np.float32)
    for x, (lo, hi) in zip(starts, owned_spans(starts, 128, 700), strict=True):
        expected[lo:hi] = (np.arange(lo, hi) - x) / 128
    assert n == 3 * 6 and prob.shape == (300, 700)
    assert torch.allclose(prob, torch.sigmoid(torch.from_numpy(expected)).expand(300, 700))


@pytest.mark.parametrize('scale', [1.0, 0.5])
def test_odd_sized_photo_scores_every_native_pixel(scale):
    photo = _photo(293, 417, scale)
    cfg = Config(window=128, scale=scale, amp=False, batch_size=4)
    s = score(Constant(-1.0), [photo], cfg, CPU, thresholds=(0.25, 0.5))
    assert to_native(torch.zeros(146, 208), (293, 417)).shape == (293, 417)
    assert s.hist.pos.sum() + s.hist.neg.sum() == 293 * 417
    assert s.hist.pos.sum() == photo.mask.sum()
    for t in (0.25, 0.5):  # per-photograph counts add up to the pooled ones
        assert tuple(s.photo_counts[t].sum(0)) == tuple(int(v) for v in s.hist.counts(t))
    assert s.photo_counts[0.25][0].tolist() == [photo.mask.sum(), (~photo.mask).sum(), 0, 0]


def test_window_loss_is_averaged_over_windows():
    photo = _photo(256, 384, 1.0)
    s = score(Constant(0.0), [photo], Config(window=128, amp=False), CPU,
              criterion=nn.BCEWithLogitsLoss())  # fmt: skip
    assert s.loss == pytest.approx(np.log(2), rel=1e-6)


@pytest.mark.parametrize('scale', [1.0, 0.5])
def test_paper_patches_of_a_photograph(scale):
    photo = _photo(300, 520, scale, seed=7)  # half size 150 x 260: one row of two patches
    cfg = Config(window=128, scale=scale, amp=False)
    s = score(Constant(5.0), [photo], cfg, CPU, thresholds=(0.5,), paper=True)
    assert s.paper_index == [('7', 0, 0), ('7', 0, 1)]
    counts = s.paper_counts[0.5]
    half = resize_mask(photo.mask, 0.5)
    for k, (_, _, c) in enumerate(s.paper_index):
        rust = int(half[:128, 128 * c : 128 * (c + 1)].sum())
        assert counts[k].tolist() == [rust, 128 * 128 - rust, 0, 0]  # everything predicted rust
    assert s.paper_rust.tolist() == [bool(r) for r in counts[:, 0] > 0]


def test_paper_order_sorts_photographs_as_text():
    assert paper_order(['25', '105', '29', '123']) == ['105', '123', '25', '29']


def test_authors_average_matches_scikit_learn_per_batch():
    metrics = pytest.importorskip('sklearn.metrics')
    rng = np.random.default_rng(0)
    pred = rng.random((23, 16)) > 0.7
    true = rng.random((23, 16)) > 0.8
    pred[5:10] = true[5:10] = False  # a batch without any rust: every score divides by zero
    counts = np.stack([[(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()]
                       for p, t in zip(pred, true, strict=True)])  # fmt: skip
    expected = {'Precision': 0.0, 'Recall': 0.0, 'F1': 0.0}
    for i in range(0, 23, 5):
        y, x = true[i : i + 5].ravel(), pred[i : i + 5].ravel()
        n = len(true[i : i + 5])
        expected['Precision'] += n * metrics.precision_score(y, x, zero_division=0)
        expected['Recall'] += n * metrics.recall_score(y, x, zero_division=0)
        expected['F1'] += n * metrics.f1_score(y, x, zero_division=0)
    got = authors_average(counts, batch=5)
    assert got == pytest.approx({k: v / 23 for k, v in expected.items()})


def test_rust_only_scores_batch_after_the_removal():
    counts = np.array([[5, 1, 1, 9], [0, 4, 0, 12], [1, 0, 6, 9], [0, 0, 0, 16]])
    rust = np.array([True, False, True, False])
    s = paper_scores(counts, rust, batch=2)
    assert s['authors_rust_only'] == authors_average(counts[rust], batch=2)
    assert s['authors_rust_only'] != pytest.approx(authors_average(counts, batch=2))
    assert s['rust_only']['IoU'] == pytest.approx(6 / (6 + 1 + 7))
    assert s['pooled']['Precision'] == pytest.approx(6 / 11)
