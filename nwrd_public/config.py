"""Experiment protocol. Every model is trained and scored with exactly these settings.

Each field is also a command-line flag of `python -m nwrd_public.run` (`data_root` ->
`--data-root`). A preset sets the window size and scale; flags given explicitly override it.
"""

from dataclasses import asdict, dataclass, field

# CANet-B4 is ours; UNet is the NWRD paper's architecture; DeepLabV3+ R50 is the second baseline.
# models.py, copied unchanged from NWRD_final, also defines that study's `unet_tuned` control and
# `deeplabv3plus_b4` ablation (and mentions its config.unet_tuned_lr); they are not run here.
COMPARED_MODELS = ('canet_b4', 'unet', 'deeplabv3plus_r50')


@dataclass
class Config:
    data_root: str = ''  # holds the release's train/ and test/; '' searches /kaggle/input
    manifest: str = ''  # MD5 manifest of the release; '' is data/manifest.json
    out_dir: str = 'runs'
    model: str = 'canet_b4'
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    data_seed: int = 42  # background subsample; fixed so all runs see the same windows

    # Windows of `window` px are cut from each photograph resized by `scale` (Lanczos), on the
    # fitted grid of nwrd_public/windows.py: no padding, every pixel covered.
    window: int = 512
    scale: float = 1.0
    window_dir: str = ''  # where training windows are cut; default: under the temp directory
    bg_keep_ratio: float = 0.20
    epochs: int = 30
    freeze_encoder_epochs: int = 3
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 5e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    dice_weight: float = 0.7
    bce_weight: float = 0.3
    dropout: float = 0.5  # CANet head dropout; the other models have none
    sched_t0: int = 5
    sched_tmult: int = 2
    sched_eta_min: float = 1e-7
    amp: bool = True
    pretrained: bool = True

    # Threshold chosen on val over this grid, then frozen and applied to test. Rust is sparse in
    # this release, so the BCE rust weight is high and the best threshold may lie above 0.95.
    thr_lo: float = 0.01
    thr_hi: float = 0.99
    thr_step: float = 0.01

    time_budget_hours: float = 0.0  # >0: stop cleanly before a platform session limit
    limit: int = 0  # >0: use only the first N photographs of each split (smoke tests)

    def validate(self) -> None:
        if self.model not in COMPARED_MODELS:
            raise ValueError(f'model {self.model!r} is not compared here; use {COMPARED_MODELS}')
        if not 0 < self.scale <= 1:
            raise ValueError(
                f'scale {self.scale} must be in (0, 1]: photographs are never enlarged'
            )
        if self.window <= 0:
            raise ValueError(f'window {self.window} must be positive')

    def to_dict(self) -> dict:
        return asdict(self)


# The two scales. `512px` is native resolution; `128px` is the NWRD paper's scale, 128px windows
# of photographs halved with Lanczos. Everything else is the same, so 512px vs 128px is a
# one-variable comparison.
PRESETS = {
    '512px': {'window': 512, 'scale': 1.0},
    '128px': {'window': 128, 'scale': 0.5},
}


def preset_of(window: int, scale: float) -> str:
    """The preset with this window size and scale."""
    for name, p in PRESETS.items():
        if p['window'] == window and p['scale'] == scale:
            return name
    raise ValueError(f'no preset has window {window} and scale {scale}')
