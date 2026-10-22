# Suggestions

## Open

### Report the annotation-agreement ceiling beside the models
**What:** In `docs/results.md`, put the twin masks' agreement (IoU 0.40–0.63, median 0.50, from
`results/dataset/duplicates.json`) beside each model's test IoU, and per test photograph where
a twin exists (80, 89).
**Why it's worth it:** Absolute IoUs here will be low; without the human ceiling a reader
compares them with NWRD_final's 0.73 and misreads them.
**Cost:** A paragraph and a small table when writing `docs/results.md`.
**Why not yet:** Needs the Kaggle results.

### Qualitative prediction figures for a few test photographs
**What:** A notebook stage that saves overlays (image, mask, prediction at the run's threshold)
for three test photographs of different cameras, for one seed of each model.
**Why it's worth it:** Reviewers ask to see what the models get wrong; NWRD_final had such a
figure. The weights stay on Kaggle, so this must run inside the kernel.
**Cost:** A function in `evaluate.py` plus a stage in `run.py`; the notebooks rebuild.
**Why not yet:** Not in the approved plan; decide before the first Kaggle push, since adding it
later means re-running or keeping `best.pt` downloads.

## Rejected

### Capping `pos_weight` (e.g. at 10) for the sparse rust — 2026-10-22
D8 kept NWRD_final's rule (63.1 / 43.9): a cap would be a new, untested setting chosen without
evidence; the threshold grid was widened instead.

### Re-running speed benchmarks — 2026-10-22
D5: speed depends on architecture and window size, not on the photographs; NWRD_final's T4
measurements are reused (`results/nwrd_final/`).

## Implemented
