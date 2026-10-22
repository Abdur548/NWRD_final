"""Runs one model at one scale: cut -> train -> eval -> summary.

    python -m nwrd_public.run --model canet_b4 --preset 512px --data-root data/raw/NWRD --out runs
    python -m nwrd_public.run --model unet --preset 128px --out runs   # on Kaggle

On Kaggle the release is found under /kaggle/input. It is checked against data/manifest.json
before anything else, and the run stops if any file differs. Finished stages are skipped, and an
interrupted training run resumes from last.pt, so the same command can be repeated across Kaggle
sessions until everything is done. Exit code 3 means the time budget stopped training: run the
same command again to continue.

The side-by-side comparison of the three models: python -m nwrd_public.compare results
"""

import argparse
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import traceback
import typing
from dataclasses import fields

from . import __version__
from .config import PRESETS, Config

STAGES = ('cut', 'train', 'eval', 'summary')
# Stages that create run records. Only these write config.json and log.txt, so re-writing a
# finished run's summary never overwrites what the run recorded.
RECORDING_STAGES = {'cut', 'train', 'eval'}
# Settings that may change between the sessions of one experiment: where things are, how many
# workers, the time budget and which seeds this session runs. Every other setting must match the
# experiment's config.json, since a seed's folder does not record its window or scale.
SESSION_FIELDS = {
    'data_root', 'manifest', 'out_dir', 'window_dir', 'num_workers', 'time_budget_hours', 'seeds',
}  # fmt: skip


def _parse_bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ('1', 'true', 'yes'):
        return True
    if v in ('0', 'false', 'no'):
        return False
    raise argparse.ArgumentTypeError(f'expected true or false, got {s!r}')


def parse_args(argv=None) -> argparse.Namespace:
    """Every Config field is a flag; the preset's values are the defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--preset', choices=sorted(PRESETS), default='512px')
    preset = pre.parse_known_args(argv)[0].preset
    defaults = {**Config().to_dict(), **PRESETS[preset]}
    hints = typing.get_type_hints(Config)

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        '--preset',
        choices=sorted(PRESETS),
        default='512px',
        help='the scale whose window size and resize factor are the defaults (config.py)',
    )
    p.add_argument('--stage', default='all', choices=('all',) + STAGES)
    p.add_argument('--out', dest='out_dir', default=defaults['out_dir'])
    for f in fields(Config):
        if f.name == 'out_dir':
            continue
        flag, hint, default = '--' + f.name.replace('_', '-'), hints[f.name], defaults[f.name]
        if hint is bool:
            p.add_argument(
                flag, dest=f.name, type=_parse_bool, default=default, metavar='{true,false}'
            )
        elif typing.get_origin(hint) is list:
            p.add_argument(
                flag, dest=f.name, nargs='+', type=typing.get_args(hint)[0], default=default
            )
        else:
            p.add_argument(flag, dest=f.name, type=hint, default=default)
    return p.parse_args(argv)


def _environment() -> str:
    """Library versions and device, recorded at the top of every session's log."""
    from importlib.metadata import PackageNotFoundError, version

    import cv2
    import numpy
    import torch

    def v(package):
        try:
            return version(package)
        except PackageNotFoundError:
            return 'not installed'

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return (
        f'python {platform.python_version()}, torch {torch.__version__}, '
        f'numpy {numpy.__version__}, opencv {cv2.__version__}, '
        f'albumentations {v("albumentations")}, '
        f'segmentation-models-pytorch {v("segmentation_models_pytorch")}, '
        f'efficientnet-pytorch {v("efficientnet_pytorch")}, device {device}'
    )


def default_window_dir(cfg: Config) -> str:
    """Training windows are cut outside out_dir, so they never enter a run's saved output."""
    return os.path.join(
        tempfile.gettempdir(), 'nwrd_public_windows', f'{cfg.window}px_x{cfg.scale:g}'
    )


def check_same_experiment(cfg: Config) -> None:
    """Refuses to continue the runs in cfg.out_dir if they were made with other settings: they
    would be skipped as finished, reported and collected as this experiment's."""
    recorded = os.path.join(cfg.out_dir, 'config.json')
    if not os.path.exists(recorded):
        return
    with open(recorded) as f:
        done = json.load(f)
    differ = {
        k: (done.get(k), v)
        for k, v in cfg.to_dict().items()
        if k not in SESSION_FIELDS and done.get(k) != v
    }
    if differ:
        raise RuntimeError(
            f'{cfg.out_dir} holds runs made with other settings (recorded, now): {differ}; '
            'use another --out'
        )


def main(argv=None) -> int:
    t_session = time.time()  # the time budget counts from here, across all runs
    a = parse_args(argv)
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
    cfg.validate()
    stages = STAGES if a.stage == 'all' else (a.stage,)
    recording = bool(RECORDING_STAGES & set(stages))
    if recording:
        check_same_experiment(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)
    log_file = (
        open(os.path.join(cfg.out_dir, 'log.txt'), 'a', encoding='utf-8') if recording else None
    )

    def log(msg: str) -> None:
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        if log_file:
            log_file.write(line + '\n')
            log_file.flush()

    try:
        return _run(cfg, a.preset, stages, log, t_session, recording)
    except Exception:
        # On Kaggle the notebook's own output cannot be downloaded, but runs/log.txt can.
        if log_file:
            log_file.write(f'{time.strftime("%H:%M:%S")} CRASHED:\n{traceback.format_exc()}')
        raise
    finally:
        if log_file:
            log_file.close()


def _run(cfg: Config, preset: str, stages: tuple, log, t_session: float, recording: bool) -> int:
    log(
        f'nwrd_public {__version__} | {cfg.model} | preset {preset} | '
        f'stages {",".join(stages)} | {_environment()}'
    )
    release = windows = None
    if {'cut', 'train', 'eval'} & set(stages):
        from .dataset import open_release

        release = open_release(cfg.data_root or None, cfg.manifest or None, cfg.limit, log)
        cfg.data_root = release.root
    if {'cut', 'train'} & set(stages):
        from .windows import cut_training_windows

        windows = cut_training_windows(
            release,
            release.splits['train'],
            cfg.window_dir or default_window_dir(cfg),
            cfg.window,
            cfg.scale,
            cfg.bg_keep_ratio,
            cfg.data_seed,
            workers=max(1, cfg.num_workers),
            log=log,
        )
        shutil.copy(
            os.path.join(windows['dir'], 'windows.json'), os.path.join(cfg.out_dir, 'windows.json')
        )
    if recording:
        with open(os.path.join(cfg.out_dir, 'config.json'), 'w') as f:
            json.dump({'preset': preset, **cfg.to_dict()}, f, indent=2)

    complete, session_start = True, None
    for seed in cfg.seeds:
        run_dir = os.path.join(cfg.out_dir, cfg.model, f'seed{seed}')
        if 'train' in stages and not os.path.exists(os.path.join(run_dir, 'TRAINED')):
            from .train import train_run

            if not train_run(
                cfg, release, windows, seed, run_dir, log=log, session_start=session_start
            ):
                complete = False
                break  # out of time budget: stop here, resume next session
            session_start = t_session
        if (
            'eval' in stages
            and os.path.exists(os.path.join(run_dir, 'TRAINED'))
            and not os.path.exists(os.path.join(run_dir, 'metrics.json'))
        ):
            from .evaluate import evaluate_run

            evaluate_run(cfg, release, run_dir, log=log)

    if 'summary' in stages:
        from .compare import write_model_summary

        path = write_model_summary(cfg.out_dir, cfg.model)
        if path:
            with open(path, encoding='utf-8') as f:
                log(f.read())

    if not complete:
        log(
            'INCOMPLETE: continue in a new session with these runs/ in place (on Kaggle: '
            'scripts/run_on_kaggle.py --resume, which attaches them).'
        )
    return 0 if complete else 3


if __name__ == '__main__':
    # The reports contain ±, which a redirected stdout on Windows (cp1252) cannot encode.
    sys.stdout.reconfigure(encoding='utf-8')
    sys.exit(main())
