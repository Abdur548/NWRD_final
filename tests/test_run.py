"""Run bookkeeping that the end-to-end tests do not isolate."""

import json

import pytest

from nwrd_public.config import Config
from nwrd_public.run import check_same_experiment, main


def _record(out, cfg: Config) -> None:
    out.mkdir(exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'preset': '512px', **cfg.to_dict()}))


def test_runs_of_other_settings_are_never_continued(tmp_path):
    out = tmp_path / 'runs'
    _record(out, Config(out_dir=str(out)))
    with pytest.raises(RuntimeError, match='other settings'):
        main(['--preset', '128px', '--out', str(out), '--stage', 'train'])
    assert [p.name for p in out.iterdir()] == ['config.json']  # nothing written, no log


def test_sessions_of_one_experiment_may_differ_in_paths_workers_budget_and_seeds(tmp_path):
    out = tmp_path / 'runs'
    _record(out, Config(out_dir=str(out), limit=2))
    same = Config(out_dir=str(out), limit=2, data_root='elsewhere', window_dir='w', num_workers=8,
                  time_budget_hours=6.0, seeds=[43])  # fmt: skip
    check_same_experiment(same)
    with pytest.raises(RuntimeError, match="'limit': \\(2, 0\\)"):
        check_same_experiment(Config(out_dir=str(out)))  # a smoke run's folder, a full run
