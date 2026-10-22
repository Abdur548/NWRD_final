"""The Kaggle notebooks must embed the current nwrd_public/ package and manifest, and differ only
in the model they train; Kaggle runs the notebook, not the repository, so a stale notebook would
silently run old code."""

import json
import os
import sys

import pytest

from .conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import build_notebooks  # noqa: E402
import run_on_kaggle  # noqa: E402

MODELS = run_on_kaggle.MODELS


@pytest.mark.parametrize('model', MODELS)
def test_notebook_is_up_to_date(model):
    path = os.path.join(ROOT, 'notebooks', f'{model}.ipynb')
    with open(path, encoding='utf-8') as f:
        on_disk = f.read()
    assert on_disk == build_notebooks.render(build_notebooks.build(model)), (
        f'{path} is stale: run python scripts/build_notebooks.py'
    )


@pytest.mark.parametrize('model', MODELS)
def test_notebooks_differ_only_in_their_model(model):
    canet = build_notebooks.render(build_notebooks.build('canet_b4'))
    text = build_notebooks.render(build_notebooks.build(model))
    assert text.count(f"MODEL = '{model}'") == 1
    assert text.replace(f"MODEL = '{model}'", "MODEL = 'canet_b4'") == canet


def test_notebook_lines_the_runner_rewrites():
    """--preset, --resume and --budget-hours each rewrite exactly one line of the pushed copy."""
    nb = build_notebooks.render(build_notebooks.build('unet'))
    for line in (run_on_kaggle.PRESET_LINE, run_on_kaggle.EXPECT_OFF, run_on_kaggle.BUDGET_LINE):
        assert nb.count(line) == 1, line


def test_notebooks_embed_the_package_and_the_manifest():
    cells = [''.join(c['source']) for c in build_notebooks.build('unet')['cells']]
    embedded = [c.split('\n', 1) for c in cells if c.startswith('%%writefile')]
    assert [head for head, _ in embedded] == ['%%writefile code/data/manifest.json'] + [
        f'%%writefile code/nwrd_public/{m}.py' for m in build_notebooks.MODULES
    ]
    package = {f[:-3] for f in os.listdir(os.path.join(ROOT, 'nwrd_public')) if f.endswith('.py')}
    assert package == set(build_notebooks.MODULES)
    with open(os.path.join(ROOT, 'data', 'manifest.json'), encoding='utf-8') as f:
        assert json.loads(embedded[0][1]) == json.load(f)
