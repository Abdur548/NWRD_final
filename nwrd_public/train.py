"""Training for one (model, seed) run. Resumable from last.pt."""

import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .dataset import Release, RustDataset, train_transform
from .evaluate import load_photos, score
from .metrics import scores
from .models import build_model


class CombinedLoss(nn.Module):
    """dice_w * Dice loss + bce_w * BCE, both on logits; BCE weights rust pixels by pos_weight."""

    def __init__(self, dice_w: float, bce_w: float, pos_weight: torch.Tensor):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.dice_w, self.bce_w = dice_w, bce_w
        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, target):
        return self.dice_w * self.dice(logits, target) + self.bce_w * self.bce(logits, target)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Seeds a DataLoader worker's augmentation from the seed PyTorch derives for that worker
    from the loader's generator, so augmentation depends on (seed, epoch, worker) only.
    Albumentations 2 seeds its transforms from the OS otherwise, not from the global seed."""
    info = torch.utils.data.get_worker_info()
    info.dataset.transform.set_random_seed(info.seed % 2**32)


def make_train_loader(cfg: Config, windows: dict):
    """The loader over the cut training windows, and its generator (reseed it every epoch)."""
    g = torch.Generator()
    loader = torch.utils.data.DataLoader(
        RustDataset(
            os.path.join(windows['dir'], 'images'),
            os.path.join(windows['dir'], 'masks'),
            windows['files'],
            train_transform(),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        generator=g,
        worker_init_fn=_seed_worker,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, g


def train_run(
    cfg: Config,
    release: Release,
    windows: dict,
    seed: int,
    run_dir: str,
    log=print,
    session_start=None,
) -> bool:
    """Trains one run to cfg.epochs on the cut windows, keeping the checkpoint with the best
    pooled IoU at threshold 0.5 over the whole validation photographs as best.pt. Returns False
    if the time budget stopped it (resume later)."""
    os.makedirs(run_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = cfg.amp and device.type == 'cuda'
    torch.backends.cudnn.benchmark = True
    set_seed(seed)

    train_loader, gen = make_train_loader(cfg, windows)
    val_photos = load_photos(release, release.splits['val'], cfg.scale)
    keys = ('windows', 'rust_windows', 'bg_kept', 'bg_total', 'sample', 'pos_weight')
    info = {**{k: windows[k] for k in keys}, 'val_photos': len(val_photos)}
    log(f'[{cfg.model} s{seed}] data: {info}')
    model = build_model(cfg.model, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    criterion = CombinedLoss(
        cfg.dice_weight, cfg.bce_weight, torch.tensor([windows['pos_weight']], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.sched_t0, T_mult=cfg.sched_tmult, eta_min=cfg.sched_eta_min
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    history = {
        k: []
        for k in (
            'train_loss',
            'val_loss',
            'val_iou',
            'val_dice',
            'val_iou_best_thr',
            'lr',
            'epoch_seconds',
        )
    }
    start_epoch, best_iou = 0, -1.0
    last_path, best_path = os.path.join(run_dir, 'last.pt'), os.path.join(run_dir, 'best.pt')
    if os.path.exists(last_path):
        # Load on the CPU: torch.set_rng_state only accepts a CPU tensor, and load_state_dict
        # moves model and optimizer state onto the model's device by itself.
        ck = torch.load(last_path, map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler'])
        scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['rng_cpu'])
        if device.type == 'cuda' and ck.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(ck['rng_cuda'])
        start_epoch, best_iou, history = ck['epoch'], ck['best_iou'], ck['history']
        log(f'[{cfg.model} s{seed}] resumed at epoch {start_epoch}')

    t_start = time.time()
    first_in_session = session_start is None
    session_start = t_start if first_in_session else session_start
    for epoch in range(start_epoch, cfg.epochs):
        # The budget covers the whole session, not this run. The session's first run always
        # gets one epoch, so every session makes progress; the headroom between the budget
        # and the platform limit covers that epoch when no per-epoch estimate exists yet.
        if cfg.time_budget_hours and (epoch > start_epoch or not first_in_session):
            done = epoch - start_epoch
            per_epoch = (time.time() - t_start) / done if done else 0.0
            if time.time() - session_start + 1.5 * per_epoch > cfg.time_budget_hours * 3600:
                log(f'[{cfg.model} s{seed}] time budget reached at epoch {epoch}; resume later')
                return False

        # Freezing only makes sense for an ImageNet-pretrained encoder.
        frozen = epoch < cfg.freeze_encoder_epochs and getattr(
            model, 'has_pretrained_encoder', True
        )
        for p in model.encoder.parameters():
            p.requires_grad = not frozen

        # Shuffle order and augmentation depend on (seed, epoch) only, so a run is reproducible
        # from its seed and a resumed run sees the batches it would have seen uninterrupted.
        # Worker processes reseed their augmentation from `gen` (_seed_worker); without
        # workers, the augmentation runs here.
        gen.manual_seed(seed * 10_000 + epoch)
        train_loader.dataset.transform.set_random_seed(seed * 10_000 + epoch)
        t0 = time.time()
        model.train()
        loss_sum, n = 0.0, 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                logits = model(imgs)
            loss = criterion(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * imgs.size(0)
            n += imgs.size(0)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Validation scores whole photographs at native resolution, as the final evaluation does.
        v = score(model, val_photos, cfg, device, criterion=criterion)
        val = scores(*v.hist.counts(0.5))
        _, sweep = v.hist.best_threshold(lo=cfg.thr_lo, hi=cfg.thr_hi, step=cfg.thr_step)
        for k, value in (
            ('train_loss', loss_sum / n),
            ('val_loss', v.loss),
            ('val_iou', val['IoU']),
            ('val_dice', val['Dice']),
            ('val_iou_best_thr', max(r['IoU'] for r in sweep)),
            ('lr', lr),
            ('epoch_seconds', time.time() - t0),
        ):
            history[k].append(value)
        log(
            f'[{cfg.model} s{seed}] ep {epoch + 1:02d}/{cfg.epochs} '
            f'{"(enc frozen) " if frozen else ""}train {loss_sum / n:.4f} val {v.loss:.4f} '
            f'IoU@0.5 {val["IoU"]:.4f} Dice@0.5 {val["Dice"]:.4f} '
            f'({history["epoch_seconds"][-1]:.0f}s)'
        )

        # Model selection: pooled val IoU at the fixed 0.5 threshold.
        if val['IoU'] > best_iou:
            best_iou = val['IoU']
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'val_iou@0.5': best_iou,
                    'model_name': cfg.model,
                    'seed': seed,
                },
                best_path,
            )
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'rng_cpu': torch.get_rng_state(),
                'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                'epoch': epoch + 1,
                'best_iou': best_iou,
                'history': history,
            },
            last_path,
        )
        with open(os.path.join(run_dir, 'history.json'), 'w') as f:
            json.dump({'history': history, 'data': info, 'best_val_iou@0.5': best_iou}, f, indent=1)

    os.remove(last_path)  # optimizer state is ~2/3 of the file; best.pt is what we keep
    with open(os.path.join(run_dir, 'TRAINED'), 'w') as f:
        f.write(f'best val IoU@0.5 {best_iou:.6f}\n')
    return True
