# Paper draft (reference)

A LaTeX reference draft of the paper: DeepLabV3+ R50 against the NWRD paper's UNet (and its
published UNet + APF) on the public NWRD release, with the architectural ablation. It is a
starting point to write the paper from, not the submission.

## On Overleaf

New Project → Upload Project → the zip. Compiler: pdfLaTeX (Overleaf's default); the
bibliography is BibTeX with `unsrtnat`. `main.tex` is the main document.

- **Red** text is a number, table or passage still to come: the ablation's results, and the
  prose on 128px, whose numbers and tables are already generated.
- **Blue** notes are for the authors; remove them before submission.

## What is generated

`generated/numbers.tex` and `generated/tab_*.tex` are written by `scripts/build_paper.py` from
`results/`, and the figures in `figures/` are copied from there. `main.tex` uses the numbers
through `\val{key}` (e.g. `\val{512.dlr.iou.pm}`), so after new runs:

    python scripts/build_paper.py --zip <where>.zip

regenerates them and the zip; nothing in the prose needs retyping. Edit `main.tex` and
`references.bib` by hand, never `generated/`. Keys are `<scale>.<model>.<measure>`, with models
`dlr` (DeepLabV3+ R50), `canet`, `unet`, and the ablation's `dlrs` (no pretraining), `unetr`
(UNet R50) and `dlb` (DeepLabV3+ B4); differences are `<scale>.<a>-<b>.diou` / `.ci` / `.p`.
