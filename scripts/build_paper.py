r"""Builds the paper draft's numbers, tables and figures from results/, and its Overleaf zip.

    python scripts/build_paper.py                 # paper/generated/ and the zip in build/
    python scripts/build_paper.py --zip PATH.zip  # the zip somewhere else

The prose is paper/main.tex and paper/references.bib, written by hand. Its numbers come from
paper/generated/numbers.tex through \val{key} and its tables from paper/generated/tab_*.tex, both
written here from results/, so new results need a rebuild, not retyping. A number without results
prints as its key in red and a table without results as a red box, so nothing missing passes
unnoticed in the PDF. Model-against-model intervals beyond CANet-B4's own (comparison.json) go to
results/<scale>/pairwise.json, from the photograph bootstrap the decision rule uses.

The ablation models (ABLATION) are not trained yet. Their names are the ones
nwrd_public/models.py is to give them; until their runs exist, their rows print as pending.
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from typing import NamedTuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # run from a checkout without installing the package

from nwrd_public.compare import (  # noqa: E402
    PAPER,
    SCALES,
    load_runs,
    paired_bootstrap,
    seed_ttest,
    seed_wins,
)
from nwrd_public.metrics import iou  # noqa: E402

PAPER_DIR = os.path.join(ROOT, 'paper')
RESULTS = os.path.join(ROOT, 'results')
DEFAULT_ZIP = os.path.join(ROOT, 'build', 'nwrd_paper_overleaf.zip')
SCALE_KEYS = {'512px': '512', '128px': '128'}
FIGURES = {  # paper/figures/<name>: its source, relative to results/
    'training_curves_512px.png': ('512px', 'figures', 'training_curves.png'),
    'training_curves_128px.png': ('128px', 'figures', 'training_curves.png'),
    'mask_101.png': ('dataset', 'mask_101.png'),
}
# LaTeX's own output, left out of the zip (Overleaf compiles from source).
LATEX_BUILD = ('.aux', '.bbl', '.blg', '.fdb_latexmk', '.fls', '.log', '.out', '.synctex.gz')
TABLES = ('main', 'paper', 'pairwise', 'photos', 'ablation')


class Arch(NamedTuple):
    key: str  # the model's part of a numbers.tex key, e.g. 512.dlr.iou
    name: str
    encoder: str
    pretraining: str
    context: str  # the context module and decoder


ARCH = {
    'deeplabv3plus_r50': Arch('dlr', 'DeepLabV3+ R50', 'ResNet-50', 'ImageNet', 'ASPP'),
    'canet_b4': Arch('canet', 'CANet-B4', 'EfficientNet-B4', 'ImageNet', 'CAM'),
    'unet': Arch('unet', 'UNet (NWRD paper)', 'plain convolutions', 'none', 'UNet decoder'),
    'deeplabv3plus_r50_scratch': Arch(
        'dlrs', 'DeepLabV3+ R50, no pretraining', 'ResNet-50', 'none', 'ASPP'
    ),
    'unet_r50': Arch('unetr', 'UNet R50', 'ResNet-50', 'ImageNet', 'UNet decoder'),
    'deeplabv3plus_b4': Arch('dlb', 'DeepLabV3+ B4', 'EfficientNet-B4', 'ImageNet', 'ASPP'),
}
MAIN = ('deeplabv3plus_r50', 'canet_b4', 'unet')
ABLATION = ('deeplabv3plus_r50_scratch', 'unet_r50', 'deeplabv3plus_b4')
REFERENCE = 'deeplabv3plus_r50'  # the ablation's differences are taken from this model
# (a, b, what a - b isolates). The first three are the main comparison, the rest the ablation.
PAIRS = (
    ('deeplabv3plus_r50', 'unet', 'DeepLabV3+ against the NWRD paper architecture'),
    ('deeplabv3plus_r50', 'canet_b4', 'DeepLabV3+ against the pre-registered candidate'),
    ('canet_b4', 'unet', 'the pre-registered comparison'),
    ('deeplabv3plus_r50', 'deeplabv3plus_r50_scratch', 'ImageNet pretraining'),
    ('deeplabv3plus_r50', 'unet_r50', 'ASPP against the UNet decoder'),
    ('deeplabv3plus_r50', 'deeplabv3plus_b4', 'ResNet-50 against EfficientNet-B4'),
    ('canet_b4', 'deeplabv3plus_b4', 'CAM against ASPP, same encoder'),
    ('unet_r50', 'unet', 'a pretrained ResNet-50 encoder in the UNet'),
)
PENDING = r'\textcolor{red}{pending}'


def num(x: float, digits: int = 4) -> str:
    return f'{x:.{digits}f}'


def signed(x: float, digits: int = 4) -> str:
    return f'${x:+.{digits}f}$'


def pm(mean_sd, digits: int = 4) -> str:
    m, s = mean_sd
    return f'{m:.{digits}f}\\,$\\pm$\\,{s:.{digits}f}'


def pval(p: float) -> str:
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def ci(lo: float, hi: float) -> str:
    return f'[{signed(lo)}, {signed(hi)}]'


def tabular(cols: str, header: list[str], rows: list) -> str:
    """A booktabs tabular; a row is a list of cells, or a string written as it is (a rule or a
    group heading, which must end its own line)."""
    lines = [
        f'\\begin{{tabular}}{{{cols}}}',
        '\\toprule',
        ' & '.join(header) + ' \\\\',
        '\\midrule',
    ]
    lines += [r if isinstance(r, str) else ' & '.join(r) + ' \\\\' for r in rows]
    return '\n'.join([*lines, '\\bottomrule', '\\end{tabular}', ''])


def pending_table(what: str) -> str:
    return f'\\pendingtable{{{what}}}\n'


def _read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _write(path: str, text: str) -> None:
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def published() -> dict[str, dict]:
    """The NWRD paper's UNet + APF scores, by table: {'t3': {...}, 't4': {...}}."""
    return {p['table'].replace('Table ', 't'): p['scores'] for p in PAPER}


def published_numbers() -> dict[str, str]:
    out = {}
    for table, scores in published().items():
        for metric, v in scores.items():
            out[f'paper.{table}.{metric.lower()}'] = num(v, 3)
    return out


def model_numbers(s: str, model: str, row: dict) -> dict[str, str]:
    """numbers.tex entries of one model at one scale, from its comparison.json summary row."""
    k = f'{s}.{ARCH[model].key}'
    t, half = row['test@thr'], row['test@0.5']
    out = {
        f'{k}.seeds': str(row['n']),
        f'{k}.iou': num(t['IoU'][0]),
        f'{k}.iou.pm': pm(t['IoU']),
        f'{k}.dice.pm': pm(t['Dice']),
        f'{k}.prec.pm': pm(t['Precision']),
        f'{k}.rec.pm': pm(t['Recall']),
        f'{k}.iouhalf.pm': pm(half['IoU']),
        f'{k}.val.iou.pm': pm(row['val@thr']['IoU']),
        f'{k}.thr': num(row['threshold'][0], 2),
    }
    paper = row.get('paper')
    if paper:
        pub = published()
        allp, rust = paper['@0.5']['authors'], paper['@0.5']['authors_rust_only']
        rust_iou = paper['@0.5']['rust_only']['IoU'][0]
        out.update(
            {
                f'{k}.pp.prec': num(allp['Precision'][0], 3),
                f'{k}.pp.rec': num(allp['Recall'][0], 3),
                f'{k}.pp.f1': num(allp['F1'][0], 3),
                f'{k}.pp.precrust': num(rust['Precision'][0], 3),
                f'{k}.pp.recrust': num(rust['Recall'][0], 3),
                f'{k}.pp.f1rust': num(rust['F1'][0], 3),
                f'{k}.pp.iourust': num(rust_iou, 3),
                f'{k}.pp.f1thr': num(paper['@thr']['pooled']['Dice'][0], 3),
                f'{k}.pp.iourustthr': num(paper['@thr']['rust_only']['IoU'][0], 3),
                f'{k}.pp.f1.vst3': signed(allp['F1'][0] - pub['t3']['F1'], 3),
                f'{k}.pp.f1rust.vst4': signed(rust['F1'][0] - pub['t4']['F1'], 3),
                f'{k}.pp.iourust.vst4': signed(rust_iou - pub['t4']['IoU'], 3),
            }
        )
    return out


def pairwise(runs: dict) -> dict:
    """Every pair of PAIRS with runs on common seeds: the photograph bootstrap of a - b, the
    seed-level t-test and the seeds a wins, as compare.significance computes them for CANet-B4."""
    out = {}
    for a, b, isolates in PAIRS:
        seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
        if not seeds:
            continue
        out[f'{a}-{b}'] = {
            'a': a,
            'b': b,
            'isolates': isolates,
            'seeds': seeds,
            'bootstrap_photo': paired_bootstrap(runs[a], runs[b], seeds),
            'seed_ttest': seed_ttest(runs[a], runs[b], seeds),
            'a_wins': seed_wins(runs[a], runs[b], seeds),
        }
    return out


def pair_numbers(s: str, pw: dict) -> dict[str, str]:
    out = {}
    for p in pw.values():
        k = f'{s}.{ARCH[p["a"]].key}-{ARCH[p["b"]].key}'
        boot = p['bootstrap_photo']['IoU']
        out[f'{k}.diou'] = signed(boot['delta'])
        out[f'{k}.ci'] = ci(*boot['ci95'])
        out[f'{k}.p'] = pval(boot['p_two_sided'])
        if p['seed_ttest']:
            out[f'{k}.ttp'] = pval(p['seed_ttest']['p'])
        out[f'{k}.wins'] = f'{p["a_wins"]} of {len(p["seeds"])}'
    return out


def delta_vs(pw: dict, ref: str, model: str) -> tuple[float, float, float] | None:
    """(delta, lo, hi) of model - ref in IoU, from whichever orientation pw holds."""
    if f'{model}-{ref}' in pw:
        b = pw[f'{model}-{ref}']['bootstrap_photo']['IoU']
        return b['delta'], *b['ci95']
    if f'{ref}-{model}' in pw:
        b = pw[f'{ref}-{model}']['bootstrap_photo']['IoU']
        return -b['delta'], -b['ci95'][1], -b['ci95'][0]
    return None


def main_table(summary: dict, order: list[str]) -> str:
    best = max(order, key=lambda m: summary[m]['test@thr']['IoU'][0])
    rows = []
    for m in order:
        r = summary[m]
        cell = pm(r['test@thr']['IoU'])
        rows.append(
            [
                ARCH[m].name,
                str(r['n']),
                f'\\textbf{{{cell}}}' if m == best else cell,
                pm(r['test@thr']['Dice']),
                pm(r['test@thr']['Precision']),
                pm(r['test@thr']['Recall']),
                pm(r['test@0.5']['IoU']),
                num(r['threshold'][0], 2),
            ]
        )
    header = ['Model', 'Seeds', 'IoU', 'Dice', 'Precision', 'Recall', 'IoU at 0.5', 'Threshold']
    return tabular('lccccccc', header, rows)


def paper_table(summary: dict, order: list[str]) -> str:
    pub = published()
    order = [m for m in order if summary[m].get('paper')]
    rows = [
        "\\multicolumn{5}{l}{\\emph{All test patches, threshold 0.5, the authors' "
        'per-batch averaging}} \\\\',
        ['UNet + APF, published (Table 3)']
        + [num(pub['t3'][x], 3) for x in ('Precision', 'Recall', 'F1')]
        + ['--'],
    ]
    for m in order:
        a = summary[m]['paper']['@0.5']['authors']
        rows.append(
            [ARCH[m].name] + [num(a[x][0], 3) for x in ('Precision', 'Recall', 'F1')] + ['--']
        )
    rows += [
        '\\midrule',
        '\\multicolumn{5}{l}{\\emph{Test patches with rust (closest to Table 4), '
        'threshold 0.5}} \\\\',
        ['UNet + APF, published (Table 4)']
        + [num(pub['t4'][x], 3) for x in ('Precision', 'Recall', 'F1', 'IoU')],
    ]
    for m in order:
        p = summary[m]['paper']['@0.5']
        a = p['authors_rust_only']
        rows.append(
            [ARCH[m].name]
            + [num(a[x][0], 3) for x in ('Precision', 'Recall', 'F1')]
            + [num(p['rust_only']['IoU'][0], 3)]
        )
    return tabular('lcccc', ['Model', 'Precision', 'Recall', 'F1', 'IoU'], rows)


def pairwise_table(pw: dict) -> str:
    rows = []
    for p in pw.values():
        boot = p['bootstrap_photo']['IoU']
        t = p['seed_ttest']
        rows.append(
            [
                f'{ARCH[p["a"]].name} $-$ {ARCH[p["b"]].name}',
                p['isolates'],
                signed(boot['delta']),
                ci(*boot['ci95']),
                pval(boot['p_two_sided']),
                pval(t['p']) if t else '--',
                f'{p["a_wins"]} of {len(p["seeds"])}',
            ]
        )
    header = [
        'Comparison',
        'Isolates',
        '$\\Delta$IoU',
        '95\\% CI',
        '$p$ (photographs)',
        '$p$ (seeds)',
        'Higher on',
    ]
    return tabular('llccccc', header, rows)


def photo_table(runs: dict, order: list[str]) -> str:
    """IoU of each test photograph at each run's threshold, averaged over seeds."""
    first = next(iter(runs[order[0]].values()))
    rust = first['counts'][:, [0, 2]].sum(1) / first['counts'].sum(1)
    per_model = {m: np.mean([iou(runs[m][s]['counts']) for s in sorted(runs[m])], 0) for m in order}
    rows = []
    for i, (pid, cam) in enumerate(zip(first['photos'], first['cameras'], strict=True)):
        rows.append(
            [str(pid), str(cam), f'{100 * rust[i]:.2f}\\%'] + [num(per_model[m][i]) for m in order]
        )
    header = ['Photograph', 'Camera', 'Rust'] + [ARCH[m].name for m in order]
    return tabular('llc' + 'c' * len(order), header, rows)


def ablation_table(summary: dict, pw: dict) -> str:
    rows = []
    for m in (*MAIN, *ABLATION):
        a = ARCH[m]
        cells = [a.name, a.encoder, a.pretraining, a.context]
        if m not in summary:
            rows.append(cells + [PENDING, PENDING])
            continue
        d = delta_vs(pw, REFERENCE, m)
        diff = (
            'reference' if m == REFERENCE else (f'{signed(d[0])} {ci(d[1], d[2])}' if d else '--')
        )
        rows.append(cells + [pm(summary[m]['test@thr']['IoU']), diff])
    header = [
        'Model',
        'Encoder',
        'Pretraining',
        'Context module',
        'Test IoU',
        f'$\\Delta$IoU vs {ARCH[REFERENCE].name} [95\\% CI]',
    ]
    return tabular('llllcc', header, rows)


def twins(results_root: str) -> tuple[str, dict[str, str]] | None:
    """The table of the release's repeated photographs and their two masks' agreement, and the
    agreement's range, from the dataset study (results/dataset/duplicates.json)."""
    dup = _read_json(os.path.join(results_root, 'dataset', 'duplicates.json'))
    if dup is None:
        return None
    pairs = [(p['ids'], p['paper_splits'], p['mask_iou']) for p in dup['identical']]
    pairs += [([p['of'], p['crop']], p['paper_splits'][::-1], p['mask_iou']) for p in dup['crops']]
    # Pairs that cross the paper's splits first: those leak into its evaluation.
    pairs.sort(key=lambda p: (p[1][0] == p[1][1], int(p[0][0])))
    # At the three decimals the study stores: two would round 0.475 down to 0.47 (binary floats).
    rows = [[' / '.join(ids), ' / '.join(splits), num(m, 3)] for ids, splits, m in pairs]
    table = tabular('lcc', ['Pair', "The paper's splits", 'Mask IoU'], rows)
    agreement = np.array([p[2] for p in pairs])
    numbers = {
        'data.twins.n': str(len(pairs)),
        'data.twins.min': num(agreement.min(), 2),
        'data.twins.median': num(float(np.median(agreement)), 2),
        'data.twins.max': num(agreement.max(), 2),
    }
    return table, numbers


def build(results_root: str, paper_dir: str) -> dict:
    """Writes paper_dir/generated/ and paper_dir/figures/ from results_root. Returns
    {scale: pairwise} for the scales that have results."""
    gen = os.path.join(paper_dir, 'generated')
    os.makedirs(gen, exist_ok=True)
    numbers = published_numbers()
    tw = twins(results_root)
    _write(os.path.join(gen, 'tab_twins.tex'), tw[0] if tw else pending_table('dataset study'))
    if tw:
        numbers.update(tw[1])
    out = {}
    for scale in SCALES:
        s = SCALE_KEYS[scale]
        scale_dir = os.path.join(results_root, scale)
        comparison = _read_json(os.path.join(scale_dir, 'comparison.json'))
        if comparison is None:
            for name in TABLES:
                _write(os.path.join(gen, f'tab_{name}_{s}.tex'), pending_table(f'{scale} results'))
            continue
        runs = {m: r for m, r in load_runs(scale_dir).items() if m in ARCH}
        summary = {m: r for m, r in comparison['summary'].items() if m in ARCH}
        order = [m for m in (*MAIN, *ABLATION) if m in summary]
        pw = pairwise(runs)
        for m in order:
            numbers.update(model_numbers(s, m, summary[m]))
        numbers.update(pair_numbers(s, pw))
        tables = {
            'main': main_table(summary, order),
            'paper': paper_table(summary, order),
            'pairwise': pairwise_table(pw),
            'photos': photo_table(runs, order),
            'ablation': ablation_table(summary, pw),
        }
        for name, text in tables.items():
            _write(os.path.join(gen, f'tab_{name}_{s}.tex'), text)
        out[scale] = pw
    lines = ['% Written by scripts/build_paper.py from results/: rebuild instead of editing.']
    lines += [f'\\nwrdset{{{k}}}{{{v}}}' for k, v in sorted(numbers.items())]
    _write(os.path.join(gen, 'numbers.tex'), '\n'.join(lines) + '\n')
    fig_dir = os.path.join(paper_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    for name, src in FIGURES.items():
        path = os.path.join(results_root, *src)
        if os.path.exists(path):
            shutil.copyfile(path, os.path.join(fig_dir, name))
    return out


def zip_bundle(paper_dir: str, zip_path: str) -> list[str]:
    """Zips paper_dir with main.tex at the zip's root, as Overleaf's "Upload project" expects,
    leaving out LaTeX's build output. Returns the names written."""
    names = []
    os.makedirs(os.path.dirname(os.path.abspath(zip_path)), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(paper_dir):
            dirnames.sort()
            for f in sorted(filenames):
                if f.endswith(LATEX_BUILD) or f == 'main.pdf':
                    continue
                path = os.path.join(dirpath, f)
                name = os.path.relpath(path, paper_dir).replace(os.sep, '/')
                z.write(path, name)
                names.append(name)
    return names


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--zip', default=DEFAULT_ZIP, help='where to write the Overleaf zip')
    a = ap.parse_args(argv)
    for scale, pw in build(RESULTS, PAPER_DIR).items():
        path = os.path.join(RESULTS, scale, 'pairwise.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(pw, f, indent=1)
        print(f'wrote {path}')
    names = zip_bundle(PAPER_DIR, a.zip)
    print(f'{len(names)} files -> {a.zip}')


if __name__ == '__main__':
    main()
