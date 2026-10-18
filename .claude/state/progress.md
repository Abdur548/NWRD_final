# Progress

**Status:** NWRD_final is built, verified and pushed (2026-10-18) to the public repository
`github.com/Abdur548/NWRD-Wheat-Rust`. Both experiments are complete with all three models and
documented in `docs/results.md`. The DeepLabV3+ 128px update is committed locally, not yet
pushed: push only when the user says so.
**Updated:** 2026-10-18
**Verified by:** `python -m pytest` (77 passed, 4 min 44 s, 2026-10-18), `ruff check` /
`ruff format --check`, `npm run lint` / `npm run build` (2026-10-18) / `npm audit`
(0 vulnerabilities), a real-data smoke test of the API, the dashboard (Streamlit AppTest) and the
React UI in Chrome.

## Done (2026-10-17/18)

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

## Next

1. Push the local commits when the user asks.
2. Optional, needs the user: a LICENSE (MIT or Apache-2.0), a CITATION.cff; re-running 512px
   DeepLabV3+ (it already has 3 seeds) only if they want it.
