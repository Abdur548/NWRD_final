# Progress

**Status:** 512px (v2) and 128px (v1) complete for the three compared models (code `c46a371`),
collected and committed (`b20edb3`); the paper draft committed (`9994aac`). The paper is built
around **DeepLabV3+ R50** (user, 2026-11-15, reaffirmed 2026-11-16). **Ablation (512px):**
scratch and unet_r50 finished and fetched into `runs_kaggle/512px/<model>/abdur548_v1/`, not yet
collected; deeplabv3plus_b4 v1 pushed 2026-11-17 23:33, QUEUED on Kaggle at 23:40. All three run
the code of commit `8a42e2d` (pushed from the working tree before it was committed; launcher code
hash 507735e6). No git remote yet: nothing pushed (see Open questions).
**Updated:** 2026-11-17
**Verified by:** full suite `python -m pytest` 167 passed 2026-11-16 (log
`build/pytest_full_2026-11-16.log`); after the docs, `pytest -m "not slow"` 155 passed
2026-11-17; `ruff check` / `ruff format --check` clean.

## Working

- The release in `data/raw/NWRD` matches `data/manifest.json` (200 files).
- `dataset.splits`: validation = `PAPER_VAL_IDS` without 101; 101 is in no split.
- Validation: 563 windows (116 with rust) at 512px, 2,252 (155) at 128px; training 2,204 / 6,282
  windows per epoch, `pos_weight` 63.1 / 43.9.
- `run_on_kaggle.push` stops when Kaggle refuses a push (the CLI exits 0 on "Kernel push error").
- Navigation docs (2026-11-17, user's request): `docs/codebase.md` (the map) and READMEs in
  `nwrd_public/`, `scripts/`, `notebooks/`, `tests/`; README leads with DeepLabV3+.

## Results so far

- **512px** (test IoU, 3 seeds): DeepLabV3+ R50 0.4572 ± 0.0084, CANet-B4 0.4235 ± 0.0072, UNet
  0.3983 ± 0.0119. DeepLabV3+ − UNet +0.0589, CI [+0.0415, +0.0797], 3/3 seeds, seed t-test p
  0.037 (`results/512px/pairwise.json`). Pre-registered rule: CANet-B4 worse than DeepLabV3+,
  comparable to UNet.
- **128px**: DeepLabV3+ 0.4098 ± 0.0150, CANet-B4 0.4073 ± 0.0127, UNet 0.4011 ± 0.0365; no pair
  differs (DeepLabV3+ − UNet +0.0086, CI [−0.0225, +0.0322]).
- **Ablation, fetched but not collected** (from `--status` output, 2026-11-17):
  `deeplabv3plus_r50_scratch` 0.3816 ± 0.0293 (paper-patch F1 0.502), `unet_r50` 0.4561 ±
  0.0089 (F1 0.574). Read with care until b4 lands and `build_paper.py` gives intervals: the
  UNet decoder on the ImageNet R50 encoder is about as good as DeepLabV3+ R50, and removing
  pretraining costs about 0.076 IoU, so the gain over the paper's UNet looks like the
  pretrained encoder, not ASPP.

## Never collect these runs

- `runs_kaggle/512px/{canet_b4,deeplabv3plus_r50}/abdur548_v1/` (validated on photo 101).
- `runs_kaggle/512px/deeplabv3plus_r50/abdur548_v3/`: an accidental replicate of v2, same code
  and seeds, per-seed test IoU 0.4277 / 0.4772 / 0.4329 against v2's 0.4477 / 0.4636 / 0.4602
  (`cudnn.benchmark = True` likely). Disclose it in the paper and `docs/results.md`; don't change
  the code mid-study.

## Next

1. When b4 finishes: `run_on_kaggle.py --model deeplabv3plus_b4 --status`, then
   `collect_results.py` for each of the three ablation folders, then
   `python scripts/build_paper.py --zip "C:/Users/Abdur Rahman/Downloads/NWRD_paper_overleaf.zip"`,
   tests, commit.
2. Write the paper's `\pending` passages (128px; the ablation) and `docs/results.md`, with the
   annotation-agreement ceiling beside the IoUs and the v3 replicate disclosed.
3. Push to GitHub once the user names or creates the repository.

## Open questions

- Git remote: none. `gh` is not logged in; the credential helper is Git Credential Manager (works
  for `github.com/Abdur548/NWRD_final`). Needs: the user to create the repository (and choose
  private or public) or log in with `gh auth login`.
- Qualitative prediction figures: `best.pt` stays in each kernel's output (`--status
  --with-checkpoints`). Needs: user.
