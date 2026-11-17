"""Model definitions (NWRD_final's, plus the ablation's variants): output shapes at both window
sizes, CANet at 128px, which encoders are frozen, and safe checkpoint loading."""

import pytest
import torch
import torch.nn.functional as F

from nwrd_public.config import ABLATION_MODELS, COMPARED_MODELS
from nwrd_public.models import MODELS, ContextFlow, build_model, load_checkpoint


@pytest.mark.parametrize('name', sorted(MODELS))
def test_models_output_logits_at_input_size(name):
    m = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 128, 128)).shape == (1, 1, 128, 128)


@pytest.mark.parametrize('name', COMPARED_MODELS + ABLATION_MODELS)
def test_models_run_here_work_at_both_window_sizes(name):
    m = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 3, 512, 512)).shape == (1, 1, 512, 512)
        assert m(torch.randn(2, 3, 128, 128)).shape == (2, 1, 128, 128)


def test_only_an_imagenet_encoder_is_frozen():
    """train.py freezes `.encoder` for the first epochs unless `has_pretrained_encoder` is
    False: the from-scratch ablation must train its random encoder from the first epoch."""
    scratch = build_model('deeplabv3plus_r50_scratch', pretrained=True)  # never downloads
    assert scratch.has_pretrained_encoder is False
    for name in ('deeplabv3plus_r50', 'unet_r50', 'deeplabv3plus_b4'):
        assert build_model(name, pretrained=False).has_pretrained_encoder is False
    assert not build_model('unet').has_pretrained_encoder


def test_ablation_variants_change_one_part_of_deeplabv3plus_r50():
    import segmentation_models_pytorch as smp

    ref = build_model('deeplabv3plus_r50', pretrained=False)
    scratch = build_model('deeplabv3plus_r50_scratch')
    unet_r50 = build_model('unet_r50', pretrained=False)
    b4 = build_model('deeplabv3plus_b4', pretrained=False)
    assert type(scratch) is type(ref) is type(b4) is smp.DeepLabV3Plus
    assert isinstance(unet_r50, smp.Unet)
    assert type(scratch.encoder) is type(unet_r50.encoder) is type(ref.encoder)
    assert type(b4.encoder) is not type(ref.encoder)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match='unknown model'):
        build_model('canet_b5')


def test_canet_trains_and_predicts_at_128px():
    m = build_model('canet_b4', pretrained=False)
    m.train()
    out = m(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 1, 128, 128)
    out.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_pool_cap_leaves_512px_unchanged(monkeypatch):
    """Capping CANet's context pooling at the feature-map size (needed at 128px) must not
    change any output at the 512px scale."""

    def uncapped(self, x):  # ContextFlow.forward before the cap
        h, w = x.shape[-2:]
        xd = F.avg_pool2d(x, self.scale) if self.scale > 1 else x
        f = self.decode(self.encode(xd))
        if self.scale > 1:
            return F.interpolate(f, (h, w), mode='bilinear', align_corners=False)
        return f

    m = build_model('canet_b4', pretrained=False).eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        capped = m(x)
        monkeypatch.setattr(ContextFlow, 'forward', uncapped)
        assert torch.equal(capped, m(x))


def test_checkpoints_round_trip_with_safe_loading(tmp_path):
    m = build_model('unet', pretrained=False)
    path = tmp_path / 'best.pt'
    torch.save({'model': m.state_dict(), 'epoch': 3, 'model_name': 'unet', 'seed': 1}, path)
    n = build_model('unet', pretrained=False)
    ckpt = load_checkpoint(n, str(path))
    assert ckpt['epoch'] == 3
    a, b = m.state_dict(), n.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoints_with_pickled_code_are_refused(tmp_path):
    class Payload:
        def __reduce__(self):
            return (print, ('this would run on load',))

    path = tmp_path / 'evil.pt'
    torch.save({'model': {}, 'payload': Payload()}, path)
    with pytest.raises(Exception, match='[Ww]eights only load failed'):
        load_checkpoint(build_model('unet', pretrained=False), str(path))
