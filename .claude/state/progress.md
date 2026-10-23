# Progress

**Status:** NWRD_final is built, verified and pushed. Both experiments are complete with all
three models and documented in `docs/results.md`. On 2026-10-23 (user: "push at
github.com/Abdur548/NWRD_final") the data-provenance fix was committed and `main` pushed to the
new public repository `github.com/Abdur548/NWRD_final` (remote `final`). The old remote `origin`
(`Abdur548/NWRD-Wheat-Rust`, last pushed 2026-10-18) is behind; push there only if asked.
**Updated:** 2026-10-23
**Verified by:** `python -m pytest` (82 passed, 10 min 14 s, 2026-10-23), `ruff check` / `ruff format
--check`, `npm run lint` / `npm run build` (2026-10-23), `scripts/verify_dataset.py` on the real source
(`--counts`: 9,600 patches; `--patches --photos 91`: 192 files; 0 problems), links in the docs
(53, 0 broken), the PDFs re-rendered.

## Done (2026-10-17/19)

- Clean package `nwrd/` with presets, safe checkpoint loading, seeded augmentation, verdict in
  reports, export as its own stage, shared whole-photograph inference.
- Equivalence gates against the old code: training (163 checks), models, aggregation: identical.
- Results imported: `results/512px/{final,replicate}`, `results/128px/final`; comparisons
  regenerated (numbers identical, verdicts added); analysis for both scales; `scales.md`.
- App: FastAPI backend (limits, CORS for localhost and 127.0.0.1, EXIF, threadpool), Streamlit
  dashboard, React UI; v4 export (CANet-B4 seed 43, chosen on validation) as the served model.
- Docs consolidated to `docs/{protocol,results,paper_summary}.md`; README; PDFs render.
- DeepLabV3+ R50 at 128px (3 seeds, Kaggle version 3), imported and documented; per-model
  summaries (`<model>/summary.md`, `docs/results.md` §8); 128px benchmark re-measured.
- Fixed on the way: the Kaggle CLI's empty kernel log on Windows, and `python -m nwrd.run`
  crashing when its stdout is redirected on Windows.
- 2026-10-19, data provenance (user: "fix"): `docs/protocol.md` §3 rewritten (3.1 photographs,
  3.2 patch recipe, 3.3 splits); `scripts/verify_dataset.py` + tests; README, results, paper
  summary, app texts, `nwrd/` docstrings and the report notes no longer call the photographs
  NWRD; per-photograph cost now for 6016×4000 (96 patches; `scales.md` regenerated). 2026-10-20
  ("check kaggle"): the source is the public `manan369/nwrd-f`, identical to the private copy.

## Next

1. Commit and push further changes only when the user asks (to `final`).
2. Ask the user where NWRD-F's photographs were taken (its Kaggle page does not say), and whether
   the public Kaggle dataset `abdur548/nwrd-patched` may stay public without a license.
3. Optional, needs the user: a LICENSE (MIT or Apache-2.0), a CITATION.cff.
