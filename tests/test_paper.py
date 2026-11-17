"""The paper draft: every number, table, figure and citation paper/main.tex uses exists or is
waiting for runs, and the model differences it reports match the decision rule's own."""

import json
import os
import re
import shutil
import sys

import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import build_paper  # noqa: E402

RESULTS = os.path.join(ROOT, 'results')
PAPER = os.path.join(ROOT, 'paper')
needs_results = pytest.mark.skipif(
    not os.path.exists(os.path.join(RESULTS, '512px', 'comparison.json')),
    reason='needs results/512px',
)


def _read(path: str) -> str:
    with open(path, encoding='utf-8') as f:
        return f.read()


def _tex(path) -> str:
    """A LaTeX file without its comments, which never print."""
    return re.sub(r'(?<!\\)%.*', '', _read(path))


def _summary(scale: str) -> dict | None:
    path = os.path.join(RESULTS, scale, 'comparison.json')
    return json.loads(_read(path))['summary'] if os.path.exists(path) else None


def _awaits_runs(key: str) -> bool:
    """A key of a scale without results, or of a model not run at that scale. An unknown scale
    or model key is a typo, not a pending result."""
    parts = key.split('.')
    scale = {v: k for k, v in build_paper.SCALE_KEYS.items()}.get(parts[0])
    by_key = {a.key: m for m, a in build_paper.ARCH.items()}
    models = parts[1].split('-') if scale and len(parts) > 1 else []
    if not models or any(k not in by_key for k in models):
        return False
    summary = _summary(scale)
    return summary is None or any(by_key[k] not in summary for k in models)


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    paper = tmp_path_factory.mktemp('paper')
    for name in ('main.tex', 'references.bib'):
        shutil.copyfile(os.path.join(PAPER, name), paper / name)
    return paper, build_paper.build(RESULTS, str(paper))


@needs_results
def test_every_number_in_the_text_is_generated_or_awaits_runs(built):
    paper, _ = built
    defined = set(re.findall(r'\\nwrdset\{([^}]*)\}', _read(paper / 'generated' / 'numbers.tex')))
    used = set(re.findall(r'\\val\{([^}]*)\}', _tex(paper / 'main.tex')))
    assert {'512.dlr.iou', '512.dlr-unet.ci', 'paper.t3.f1'} <= used & defined
    assert {k for k in used - defined if not _awaits_runs(k)} == set()


def test_a_misspelt_key_is_not_taken_for_a_pending_one():
    assert not _awaits_runs('512.dlrr.iou')
    assert not _awaits_runs('paper.t5.f1')
    assert _awaits_runs('512.dlrs.iou') == (
        _summary('512px') is None or 'deeplabv3plus_r50_scratch' not in _summary('512px')
    )


@needs_results
def test_every_input_figure_and_citation_exists(built):
    paper, _ = built
    tex = _tex(paper / 'main.tex')
    for name in re.findall(r'\\input\{([^}]*)\}', tex):
        assert (paper / f'{name}.tex').exists(), name
    for name in re.findall(r'\\figureorpending\{([^}]*)\}', tex):
        assert name in build_paper.FIGURES, name
    cited = {k.strip() for g in re.findall(r'\\cite[pt]?\{([^}]*)\}', tex) for k in g.split(',')}
    entries = set(re.findall(r'@\w+\{([^,\s]+),', _read(paper / 'references.bib')))
    assert cited and cited <= entries, cited - entries


@needs_results
def test_differences_are_the_decision_rules_own(built):
    _, pairwise = built
    pw = pairwise['512px']
    rule = json.loads(_read(os.path.join(RESULTS, '512px', 'comparison.json')))['significance']
    # CANet-B4 minus UNet, oriented as the rule computes it.
    assert pw['canet_b4-unet']['bootstrap_photo'] == rule['unet']['bootstrap_photo']
    # DeepLabV3+ minus CANet-B4 is the rule's CANet-B4 minus DeepLabV3+, negated.
    ours = pw['deeplabv3plus_r50-canet_b4']['bootstrap_photo']['IoU']
    theirs = rule['deeplabv3plus_r50']['bootstrap_photo']['IoU']
    assert ours['delta'] == pytest.approx(-theirs['delta'])
    assert ours['ci95'] == pytest.approx([-theirs['ci95'][1], -theirs['ci95'][0]])
    assert pw['deeplabv3plus_r50-canet_b4']['a_wins'] == 3 - rule['deeplabv3plus_r50']['canet_wins']


def test_without_results_every_table_is_pending(tmp_path):
    paper = tmp_path / 'paper'
    assert build_paper.build(str(tmp_path / 'no_results'), str(paper)) == {}
    for f in (paper / 'generated').glob('tab_*.tex'):
        assert _read(f).startswith('\\pendingtable{'), f.name
    keys = re.findall(r'\\nwrdset\{([^}]*)\}', _read(paper / 'generated' / 'numbers.tex'))
    assert keys and all(k.startswith('paper.') for k in keys)


def test_the_zip_has_main_tex_at_its_root_and_no_latex_output(tmp_path):
    paper = tmp_path / 'paper'
    (paper / 'generated').mkdir(parents=True)
    for name in ('main.tex', 'main.aux', 'main.pdf', 'references.bib', 'generated/numbers.tex'):
        (paper / name).write_text('x', encoding='utf-8')
    names = build_paper.zip_bundle(str(paper), str(tmp_path / 'out' / 'paper.zip'))
    assert names == ['main.tex', 'references.bib', 'generated/numbers.tex']


def test_signed_numbers_use_a_minus_sign():
    assert build_paper.signed(-0.0337) == '$-0.0337$'
    assert build_paper.signed(0.0589) == '$+0.0589$'
    assert build_paper.ci(-0.05, 0.01) == '[$-0.0500$, $+0.0100$]'
