# The public NWRD release

What the data is, how it was checked, and what it implies for patching and evaluation. Every
number here comes from `results/dataset/` (`summary.md`, `photographs.json`, `duplicates.json`),
written by `scripts/describe_data.py` from the release after checking it against
`data/manifest.json`; the earlier study's figures come from `results/nwrd_final/`. How the study
uses the data is in [`protocol.md`](protocol.md).

---

## 1. Do not use the dataset's GitHub repository

The repository the release's README points to, `github.com/dll-ncai/NUST-Wheat-Rust-Disease-NWRD`,
was compromised in October 2026. Four commits from the lab's own account (2026-10-08 and
2026-10-15) added:

- `.vscode/tasks.json`, which makes VS Code run `node public/fonts/fa-solid-400.woff2` as soon
  as the folder is opened. That "font" is JavaScript that collects credentials (`~/.aws`,
  `~/.ssh`, `~/.git-credentials`, `~/.npmrc`, `~/.pypirc`, `~/.netrc`,
  `~/.claude/.credentials.json`, `~/.gemini`, `~/.kube` and API keys in the environment), sends
  them to 193.32.204.199, and installs a cron job (tagged `#ghc2`) that runs commands from that
  server every 5 minutes;
- two GitHub Actions workflows (`audit.yml`, `security-audit.yml`) that search the repository and
  its history for keys on every push and send them to the same server.

The repository holds code only (no data); its history is clean up to commit `d07304b`
(2023-06-17). This project read the authors' code as plain text at that commit, to reproduce
their split and scoring (§6), and never cloned or ran it. Anyone who opened a clone in VS Code
since 2026-10-08 should rotate the credentials listed above and, on Linux or macOS, remove the
cron line tagged `#ghc2`.

---

## 2. Where the data comes from

| Source | What it is | Use |
|---|---|---|
| **Zenodo**, doi:10.5281/zenodo.21672928 (Anwar and Shafait, v1.0, 2023-08-04, CC BY 4.0) | `NWRD.zip`, 608,898,312 bytes, MD5 `7b9193efebe1ed8c60b10c9f05d11e14`: 200 files in `train/{images,masks}` (90 + 90) and `test/{images,masks}` (10 + 10) | The reference copy: `scripts/fetch_data.py` downloads it, checks the MD5, extracts it safely to `data/raw/NWRD/` and checks every file against `data/manifest.json` |
| **Kaggle** `bcap52/nwrd-wheat` (public) | The same 200 file names with the same byte sizes (compared 2026-10-22). Its license field says "unknown", although the data is CC BY 4.0 | What the Kaggle notebooks attach; every run checks all 200 files' MD5s first and stops on any difference |

`data/manifest.json` (in git) holds the MD5 and size of each of the 200 files, taken from the
verified archive.

---

## 3. The photographs and masks

100 photographs (JPEG) with 8-bit greyscale PNG masks, of 10 sizes. The split columns are the
NWRD paper's (§6); this study then leaves four twins out of training (§4).

| Size (W×H) | Camera | Train | Val | Test | All | 512px windows | 128px windows at 0.5× |
|---|---|---|---|---|---|---|---|
| 3456×4608 (portrait) | Huawei PAR-LX1M | 31 | 1 | 3 | 35 | 63 | 252 |
| 6016×4000 | Nikon D3200 | 25 | 2 | 4 | 31 | 96 | 384 |
| 4608×3456 | Huawei PAR-LX1M | 21 | 4 | 2 | 27 | 63 | 252 |
| 7 cropped sizes, one photograph each | Huawei (4), no EXIF (3) | 4 | 2 | 1 | 7 | 45–56 | 180–224 |
| **All** | | **81** | **9** | **10** | **100** | | |

The cropped photographs are 110 (2536×4600), 111 (4608×2909), 102 (4248×2984) and 103
(3456×3584) in training; 101 (3612×3456) and 149 (3424×3968) in validation; 105 (4604×2844) in
test. 101, 102 and 103 carry no EXIF.

- **No photograph needs rotating.** 97 have EXIF orientation 1 and 3 have no EXIF, so the
  portrait photographs are stored as portrait pixels, and every mask has its photograph's stored
  size, photo 101 aside.
- **Every mask holds only the values 0 and 255**, so binarising at > 0 (the authors) or at > 127
  (this code) gives the same masks.
- **Every mask lines up with its photograph.** In all 99 masks of their photograph's size, the
  pixels under the mask are clearly more orange than the rest: mean R − G higher by 2.9 to 48.5
  (median 23.0), measured at 1/8 scale.
- **Photo 101's mask is 19 px narrower and 21 px shorter than the photograph** (3593×3435 on
  3612×3456). Tried at every placement inside the photograph, it lines up best with the
  top-left corner: rust contrast 17.4 there, against 10.1 centred, 7.1 at the bottom right and
  9.2 stretched to the photograph's size. The unannotated right and bottom strips are therefore
  cropped off (`dataset.read_photo`), which is also what the authors' loader effectively did.

---

## 4. Repeated photographs and how consistently rust is annotated

**The release holds 90 distinct scenes, not 100.** Nine pairs of files decode to identical
pixels: the same camera and capture second; 15 and 32 are the same file, and the others differ
only in 129–150 bytes of metadata. Photo 149 is photo 99 cropped at x 24, y 0 and saved again as
JPEG: at that offset the two differ by 0.66 per colour channel on average (at most 16), and 36%
of the pixels are identical. No other photograph repeats (searched for identical pixels,
thumbnails correlated above 0.98, and crops of another photograph). **Each copy has its own mask,
and the two masks of a pair agree only at IoU 0.40 to 0.63** (median 0.50, 10 pairs): the same
rust annotated twice, differently.

| Pair | The paper's splits | Mask IoU |
|---|---|---|
| 79 / **80** | train / **test** | 0.51 |
| 86 / **89** | train / **test** | 0.43 |
| 96 / **87** | train / **val** | 0.48 |
| 99 / **149** (149 is a crop of 99) | train / **val** | 0.58 |
| 15/32, 69/141, 78/81, 83/128, 91/98, 100/137 | train / train | 0.60, 0.50, 0.52, 0.50, 0.40, 0.63 |

Under the paper's split this leaks into evaluation: test photographs 80 and 89 hold 16.9% of the
test pixels and 9.7% of the test rust, and validation photographs 87 and 149 hold 11.9% of the
validation rust, while their twins are training photographs. The paper's published test numbers
therefore carry this leak. This study leaves 79, 86, 96 and 99 out of training; the six pairs
inside training keep both copies and both masks.

The twin masks also bound what any model can score: a model's IoU against one annotation cannot
be expected to exceed how well two annotations of the same photograph agree.

---

## 5. Rust

Every photograph has some rust (none is rust-free, although the README mentions healthy images),
but it covers only 0.57% of all pixels.

| Split | Photographs | Rust share per photograph: min / median / max | Pooled |
|---|---|---|---|
| Train (the paper's) | 81 | 0.03% / 0.35% / 2.99% | 0.60% |
| Validation | 9 | 0.12% / 0.27% / 0.67% | 0.30% |
| Test | 10 | 0.08% / 0.52% / 1.24% | 0.63% |
| Train (ours, without the four twins) | 77 | 0.03% / 0.36% / 2.99% | 0.61% |

The cameras carry about the same amount (median 0.37% for the Huawei, 0.32% for the Nikon).
`NWRD_final`'s training patches were 9.8% rust (`results/nwrd_final/reference.json`), about 16
times the 0.61% here, so absolute scores here will be much lower than there, and the class
weight changes ([`protocol.md`](protocol.md) §4).

---

## 6. How the NWRD paper split, patched and scored

From the authors' code at commit `d07304b`, read as text (§1), and checked against the paper's
Table 2:

- **Test set = the release's `test` folder.** The code reads only `train/` and `test/`.
- **Validation = 9 of the 90 training photographs**, drawn by `np.random.seed(0)` over the sorted
  file names (`BaseDataLoader._split_sampler`, validation_split 0.1): photographs 2, 30, 72, 87,
  101, 118, 136, 149 and 150 (`dataset.paper_validation_ids` reproduces the draw).
- **Patches:** photographs halved with a Lanczos filter before training, then 128×128 patches on a
  grid **starting at the origin, without padding**. Training uses stride 32 (and APF); validation
  and test use stride 128, so whatever does not fill a whole patch at the right and bottom edges
  is never scored.
- **Masks:** any non-zero pixel is rust.
- **Metrics (`test.py`, `model/metric.py`):** the test patches go through in a fixed order:
  photographs by file name sorted as text (105, 123, 25, 29, 62, 63, 67, 77, 80, 89), and within a
  photograph row by row from the top left. They are batched 512 at a time without shuffling, so
  the last batch has 177. For each batch, scikit-learn's `precision_score`, `recall_score` and
  `f1_score` run on the flattened pixels with their defaults (binary, a zero division counts as
  0), and each batch's score is weighted by its number of patches. A pixel is predicted rust
  where the rust channel wins the argmax of the UNet's two outputs, i.e. at a probability above
  0.5. There is no IoU in the code. Table 4 is a separately trained UNet: for it the paper
  removed background-only patches from the training and the test set and trained at stride 128
  (paper §5), in a way the code does not show, and its IoU was computed elsewhere.
- **Normalisation:** the dataset's own mean and standard deviation.

**Table 2, recomputed from the release:**

| | The paper, 100 photographs | Recomputed | The paper, 22-photograph subset | Recomputed |
|---|---|---|---|---|
| Validation patches | 2,237 | 2,237 | 468 | 468 |
| Test patches | 2,737 | 2,737 | 703 | 468 |
| Training patches at stride 32 | 327,929 | 327,971 | 72,592 | 72,592 |

Validation and test match exactly, and so does the subset's training count. The full set's
stride-32 count is 42 patches (0.013%) above the paper's. The one number that cannot be
reproduced is the subset's 703 test patches: its two test photographs (67 and 123, named in the
README) give 468.

---

## 7. Windows and the training sample

How windows are fitted to each size, and how every pixel is scored once, is in
[`protocol.md`](protocol.md) §3. Per size class (`results/dataset/summary.md`, "Windows per
size"): 512px windows overlap by at most 22 px on the three common sizes and 72 px on photo
101's, where a padded grid would add up to 503 px of black; 128px windows of the halved
photographs overlap by at most 11 px.

| Split (ours) | Photographs | 512px windows (with rust) | 128px windows (with rust) | The paper's patches (with rust) |
|---|---|---|---|---|
| Train | 77 | 5,626 (1,349) | 22,492 (2,230) | 20,566 (2,056) |
| Validation | 9 | 619 (134) | 2,462 (188) | 2,237 (176) |
| Test | 10 | 753 (232) | 3,012 (443) | 2,737 (394) |

A window or patch has rust when its (halved, at 0.5×) mask has a rust pixel. The training sample,
every rust window plus 20% of the others, is 2,204 windows at 512px and 6,282 at 128px, with
rust on 1.56% and 2.23% of their pixels.
