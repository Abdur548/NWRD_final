# Experimental protocol: CANet-B4 against the NWRD paper's baselines on the public NWRD release

*Fixed 2026-10-22, before any run.* This document states what is compared and how, so that every
number in the results can be checked and the experiments repeated. The data are described in
[`dataset.md`](dataset.md); every number quoted from it comes from `results/dataset/`.

---

## 1. Question and models

**Question.** On the photographs of the public NWRD release, does CANet with an
ImageNet-pretrained EfficientNet-B4 encoder (CANet-B4, "ours") segment wheat stripe rust more
accurately than the architecture of the NWRD paper (Anwar et al., *The NWRD Dataset*, Sensors
23(15):6942, 2023) and than DeepLabV3+ with a ResNet-50 encoder?

| Model (`nwrd_public/models.py`) | Role | Encoder | Pretraining |
|---|---|---|---|
| `canet_b4` | **Ours** | EfficientNet-B4 | ImageNet |
| `unet` | The NWRD paper's architecture (4 encoder and 4 decoder blocks, 64 to 1024 channels), re-trained under this protocol | Plain convolution blocks | None |
| `deeplabv3plus_r50` | Second baseline: the standard DeepLabV3+ of segmentation-models-pytorch 0.5.0 | ResNet-50 | ImageNet |

The architectures and the training protocol are those of the earlier study on other photographs,
`NWRD_final` (its `docs/protocol.md` §2 and §4); `models.py` and `metrics.py` are copied from it
unchanged. Two scales are run, each with all three models and three seeds:

| Scale (`--preset`) | Input | Why |
|---|---|---|
| `512px` | 512×512 windows at native resolution | The study's main protocol |
| `128px` | 128×128 windows of each photograph halved with a Lanczos filter | The NWRD paper's scale, on the paper's own photographs |

Efficiency is not measured again: it depends on the architecture and the input size, not on the
photographs, and `NWRD_final` measured it on a Tesla T4 (copied to `results/nwrd_final/`). The
comparison converts it into time per photograph.

---

## 2. Data and split

The public NWRD release (Anwar and Shafait, Zenodo doi:10.5281/zenodo.21672928, v1.0, CC BY
4.0): 100 photographs of 10 sizes with pixel masks, in `train/` (90) and `test/` (10). Every run
first checks all 200 files against their MD5s in `data/manifest.json`, taken from the verified
Zenodo archive, and stops on any difference. On Kaggle the runs read the public copy
`bcap52/nwrd-wheat`, which has the same 200 file names and byte sizes; that MD5 check is what
establishes its contents.

Rust covers 0.57% of all pixels, and every photograph has some. The release repeats ten scenes:
nine pairs of files decode to identical pixels and photo 149 is photo 99 cropped and saved
again, each copy with its own mask ([`dataset.md`](dataset.md) §4).

| Split | Photographs | Which |
|---|---|---|
| Test | 10 | The release's `test/` folder: 25, 29, 62, 63, 67, 77, 80, 89, 105, 123. Also the NWRD paper's test set |
| Validation | 8 | 2, 30, 72, 87, 118, 136, 149, 150: the NWRD paper's, drawn by its code with NumPy seed 0 from the sorted training file names (reproduced exactly), without 101 |
| Training | 77 | The other 81 photographs of `train/` except 79, 86, 96 and 99 |

*Amended 2026-10-23, before any result:* photo 101, the paper's ninth validation photograph, is
in no split. Its mask is 19 px narrower and 21 px shorter than the photograph and lines up with
it only shifted 8 px left and 1 px up, partly past the photograph's edge
([`dataset.md`](dataset.md) §3); it was first taken to line up with the top-left corner. The
first 512px runs, which validated on it, were stopped before they finished and were not used.

Photographs 79, 86, 96 and 99 are twins of test photographs 80 and 89 and of validation
photographs 87 and 149. The paper trained on them; no model here does, so no model is validated
or tested on a photograph it was trained on. The six pairs whose two copies are both training
photographs keep both copies and both masks: two annotations of one scene. The test set is
exactly the paper's, and so are its 2,737 test patches (§3.3).

---

## 3. Windows and scoring

### 3.1 Windows fitted to each photograph

A side of L pixels gets ⌈L/P⌉ windows of P pixels, spread evenly from 0 to L − P (window i starts
at round(i·(L − P)/(n − 1)); `nwrd_public/windows.py`). Every pixel is covered and nothing is
padded; neighbouring windows overlap only by what L lacks of a whole number of windows. At 512px
that is 0 px on sides of 4608, 12–14 px on 6016 and 4000, 22 px on 3456, and at most 46 px (on
cropped photographs); at 128px at most 11 px. A padded grid would instead add up to 360 px of
black, and resizing whole photographs would shrink rust pustules a few pixels wide. A side
shorter than a window is an error, never padded.

At the `128px` scale each photograph is first resized by 0.5 with an anti-aliased Lanczos filter
(Pillow's, whose kernel widens with the reduction: the low-pass filtering the paper describes),
and its mask by area: a half-size pixel is rust when at least half of the area it replaces is.
OpenCV's `INTER_LANCZOS4`, which `NWRD_final` used, keeps a fixed kernel when shrinking and lets
fine leaf texture fold back into false patterns.

Every photograph used has a mask of its own size; reading one whose mask has another size is an
error (`dataset.read_photo`), since where such a mask belongs cannot be assumed (photo 101, §2).

### 3.2 Training windows

Every training window with rust plus 20% of the background-only windows, drawn once with the
fixed data seed 42, so every model and seed trains on the same windows. They are cut to local
disk as lossless PNGs at the start of each session; the cut is deterministic.

| Scale | Training windows | With rust | Sample per epoch | Rust share of the sample's pixels |
|---|---|---|---|---|
| 512px | 5,626 | 1,349 | 2,204 | 1.56% |
| 128px | 22,492 | 2,230 | 6,282 | 2.23% |

### 3.3 Scoring every pixel once, at native resolution

Validation and test photographs are predicted whole. Each overlap between neighbouring windows
is split at its midpoint, so every pixel takes the probability of exactly one window, the one
whose centre is nearer. At the `128px` scale the half-size probability map is resized
bilinearly to the photograph's native size. Every pixel is then scored once against the
full-size mask: no padding is scored, no edge is dropped, and both scales are scored on the same
pixels. Validation during training uses exactly this scoring.

**The paper's own test patches.** Each run is also scored on the 2,737 patches the paper's
`test.py` scores: 128×128 patches of the half-size test photographs on a grid from the origin,
whatever does not fill a whole patch at the right and bottom left out. The half-size prediction
is the `128px` model's own half-size map, and the `512px` model's native map resized by area;
masks are halved by area (the authors' halving is not in their code). Each run saves the
confusion counts of every patch, in the authors' order (photographs by file name sorted as text,
then row by row), at its threshold and at 0.5, and scores them:

- pooled over all patches, and over the patches with rust only;
- as `test.py` did: scikit-learn's precision, recall and F1 on each batch of 512 consecutive
  patches (a zero division counts as 0), weighted by the batch's size, over all patches and over
  the patches with rust (batched after the removal).

These rows put our models beside the paper's published numbers; the decision rule (§5) never
uses them. The paper's Table 3 is its APF UNet scored by `test.py` on all the patches. Its
Table 4 is a separately trained UNet (stride 128, with background-only patches removed from
training as well as test, paper §5), whose IoU `test.py` does not compute; our rust-only rows
are the closest setting, not the same experiment.

---

## 4. Training and evaluation, identical for every model

| Setting | Value |
|---|---|
| Loss | 0.7·Dice + 0.3·BCE; BCE `pos_weight` = background/rust pixel ratio of the training sample: **63.1 at 512px, 43.9 at 128px** |
| Optimiser | AdamW, lr 5e-5, weight decay 1e-4 |
| Schedule | Cosine annealing with warm restarts (T₀ 5, T_mult 2, η_min 1e-7), 30 epochs |
| Batch, precision | 8; fp16 mixed precision; gradient clipping at norm 1.0 |
| Encoder freezing | Pretrained encoders frozen for the first 3 epochs; the from-scratch UNet never |
| Augmentation | Horizontal and vertical flips, 90° rotations, brightness/contrast, elastic deformation |
| Normalisation | ImageNet mean and standard deviation |
| Model selection | The epoch with the highest pooled validation IoU at threshold 0.5 |
| Threshold | Chosen per run on **validation** (IoU-maximising, grid **0.01–0.99** in steps of 0.01), frozen, then applied once to **test**; test metrics at 0.5 are reported too |
| Metrics | IoU, Dice (= F1), precision, recall, specificity and accuracy of the rust class, **pooled over every test pixel**; per test photograph; on the paper's patches (§3.3) |
| Seeds | 42, 43, 44 |
| Hardware | One Tesla T4 on Kaggle per (model, scale); sessions capped at 11 h and resumed from checkpoints |

**What changed from `NWRD_final`, and why (fixed before any run).** Rust is about 16 times
sparser here (0.61% of the training photographs' pixels, against 9.8% of `NWRD_final`'s training
patches), so the unchanged `pos_weight` rule gives 63.1 and 43.9 where `NWRD_final` had 3.65 and
2.78 (`results/nwrd_final/reference.json`). A heavily weighted rust class pushes probabilities
up, so the best threshold may lie above 0.95; the threshold grid is therefore widened from
0.05–0.95 to 0.01–0.99. Capping `pos_weight` instead would have been a new, untested setting. A
run whose threshold still lands on the grid's edge is flagged in the reports.

Rough cost on a T4 for three seeds, from `NWRD_final`'s mean epoch times scaled to these
samples (`results/dataset/summary.md`): at 512px CANet-B4 3.0 h, UNet 6.1 h, DeepLabV3+ R50
2.0 h; at 128px 2.6, 1.4 and 1.0 h. About 16 GPU-hours in all; each (model, scale) fits in one
Kaggle session.

---

## 5. Statistics and the decision rule

**Uncertainty.** The test set is 10 photographs, and the pixels of one photograph share plant,
light and annotator. The interval is therefore a **paired bootstrap that resamples whole test
photographs** (5,000 resamples, the same draws for both models and every seed) of the
seed-averaged difference in pooled test IoU. A paired t-test over seeds, which captures
training-run variance, is reported beside it. Photographs are this study's scoring units, so
there is no patch-level interval.

**Decision rule, fixed on 2026-10-22 before any run** (`NWRD_final`'s pre-registered rule,
unchanged, including its 2026-09-28 amendment that the interval resamples photographs):

- CANet-B4 **improves on** a baseline if its mean test IoU is higher, the 95% CI of the IoU
  difference excludes 0, **and** it is higher on at least 2 of the 3 seeds.
- CANet-B4 **is worse** if the CI lies entirely below 0.
- The two are **comparable** otherwise: whenever the CI includes 0, and also when the CI lies
  above 0 but CANet-B4 is higher on fewer than 2 seeds.
- The paper's published numbers are context, never the test of the claim.

*Clarified 2026-10-22, before any run:* the last case was left implicit in `NWRD_final`'s text;
its code (`verdict`, inherited unchanged) already treats it as comparable.

Each scale is judged on its own. `nwrd_public/compare.py` applies the rule (`verdict`) and
records the verdict in each scale's `comparison.md` and `comparison.json`. Results are reported
as they come out, whichever way they go.

---

## 6. Differences from the NWRD paper's pipeline

| Aspect | NWRD paper | This protocol | Effect on comparing with the paper |
|---|---|---|---|
| Photographs and test set | The release; test = its `test/` folder | The same | Same data and test photographs |
| Training set | 81 photographs, including twins of test photographs 80 and 89 and of validation photographs 87 and 149 | 77: those four twins left out | The paper's test numbers include a leak; ours do not |
| Validation set | 9 photographs, including 101 with its mask misplaced by 8 px | 8: 101 left out | Checkpoint and threshold choice only; the test set is unchanged |
| Scale and windows | 0.5× Lanczos, 128px patches on a grid from the origin; training at stride 32 | 512px native and the paper's 0.5×/128px (anti-aliased Lanczos), both on fitted grids with no padding | The `128px` scale matches its patch size and resizing as described; the authors' resizing code is not public |
| Patch sampling | APF: rust patches plus, every 5 epochs, patches where the model produced false positives | All rust windows plus a fixed 20% of background windows | APF is not reproduced; the re-trained `unet` tests the paper's architecture, not its full method |
| Loss | Focal loss | 0.7·Dice + 0.3·weighted BCE | Same for every model here |
| Optimiser and schedule | RMSprop, lr 1e-6, 500 epochs, batch 64 | AdamW 5e-5, 30 epochs, batch 8 | 500 epochs is beyond a free GPU budget |
| Normalisation | The dataset's own mean and standard deviation | ImageNet's | The pretrained encoders expect ImageNet's |
| Threshold | Argmax of two classes (= 0.5) | Chosen on validation; 0.5 also reported | – |
| Scoring | `test.py`: per batch of 512 patches, weighted; the right and bottom remainders never scored | Every pixel once at native resolution; the paper's patches and averaging also reported | Comparable on the paper's patches (§3.3) |
| Repeats | One run | 3 seeds, photograph bootstrap, seed t-test | – |
| Hardware | 2× Tesla V100 | 1× Tesla T4 | Training times are not comparable |

---

## 7. Threats to validity

1. **Test set size.** Every test number rests on 10 photographs, and the checkpoint and threshold
   are chosen on 8.
2. **Annotation agreement.** Where the release holds two masks of the same photograph, they
   agree at IoU 0.40 to 0.63 (median 0.50). A model's IoU against one annotation is bounded by
   how consistently rust is annotated, so absolute IoUs here are low by construction and should
   be read against that agreement, not against `NWRD_final`'s.
3. **Sparse rust and a high class weight.** `pos_weight` 63 and 44 are far from the values the
   shared settings were chosen at; the threshold grid was widened for that reason (§4), but the
   protocol was not re-tuned for this data.
4. **Duplicated scenes in training.** Six scenes appear twice in training, with two masks each,
   so they weigh double.
5. **Run-to-run noise.** In `NWRD_final` the same CANet-B4 seed run in two sessions differed by
   0.028 test IoU (`results/nwrd_final/reference.json`); hence three seeds and a bootstrap.
6. **One schedule for all.** 30 epochs and shared hyperparameters suit fine-tuning pretrained
   encoders and may under-train the from-scratch UNet.
7. **APF is not reproduced**, and pretraining is part of the architectures compared.
8. **Efficiency** was measured by `NWRD_final` on one GPU type (T4).

---

## 8. Reproducing

```bash
pip install -e ".[dev]" -c constraints.txt
python scripts/fetch_data.py                       # the release, checked against data/manifest.json
python scripts/describe_data.py                    # results/dataset/ (the dataset study)
python scripts/run_on_kaggle.py --model canet_b4 --watch               # one (model, scale) on a T4
python scripts/run_on_kaggle.py --model canet_b4 --preset 128px --watch
python scripts/collect_results.py runs_kaggle/512px/canet_b4/<account>_v<N>
python -m nwrd_public.compare results              # every comparison and results/summary.csv
```

With a GPU and the release, locally:
`python -m nwrd_public.run --model canet_b4 --preset 512px --data-root data/raw/NWRD --out runs`.
