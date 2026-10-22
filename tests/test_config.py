"""The protocol's settings and the two presets."""

import pytest

from nwrd_public.config import COMPARED_MODELS, PRESETS, Config, preset_of


def test_presets_fix_window_and_scale():
    assert PRESETS == {
        '512px': {'window': 512, 'scale': 1.0},
        '128px': {'window': 128, 'scale': 0.5},
    }
    assert preset_of(512, 1.0) == '512px' and preset_of(128, 0.5) == '128px'
    with pytest.raises(ValueError):
        preset_of(256, 1.0)


def test_protocol_defaults():
    c = Config()
    assert (c.window, c.scale) == (512, 1.0)
    assert (c.thr_lo, c.thr_hi, c.thr_step) == (0.01, 0.99, 0.01)  # D8: widened grid
    assert c.seeds == [42, 43, 44] and c.data_seed == 42 and c.bg_keep_ratio == 0.2
    assert (c.epochs, c.batch_size, c.lr, c.weight_decay) == (30, 8, 5e-5, 1e-4)
    assert (c.dice_weight, c.bce_weight, c.freeze_encoder_epochs) == (0.7, 0.3, 3)


def test_only_the_compared_models_and_valid_scales_are_accepted():
    assert COMPARED_MODELS == ('canet_b4', 'unet', 'deeplabv3plus_r50')
    for m in COMPARED_MODELS:
        Config(model=m).validate()
    with pytest.raises(ValueError, match='unet_tuned'):
        Config(model='unet_tuned').validate()
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            Config(scale=bad).validate()
