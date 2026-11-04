"""Training dan evaluasi. Jalankan: ``python -m src.train``."""
import time

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .config import (CKPT_DIR, DATA_DIR, DEVICE, EPOCHS, LABEL_SMOOTHING, LR,
                     MAX_MINUTES, SEED, WEIGHT_DECAY, WORK)
from .data import build_loaders
from .model import build_model
from .utils import seed_everything


def train_one_epoch(model, loader, optimizer, scheduler, scaler):
    """Melatih model satu epoch (mixed precision).

    Returns:
        Rata-rata loss epoch.
    """
    model.train()
    total, n = 0.0, 0
    for x, y in tqdm(loader, leave=False, desc="train"):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(pixel_values=x).logits.float()
            loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); scheduler.step()
        total += loss.item() * len(y); n += len(y)
    return total / n


def fit(model, train_loader, val_loader, epochs, lr, ckpt_path, max_minutes=None):
    """Melatih beberapa epoch, menyimpan bobot dengan val accuracy terbaik.

    Returns:
        Riwayat training (list of dict).
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    history, best, t0 = [], 0.0, time.time()
    for ep in range(1, epochs + 1):
        tl = train_one_epoch(model, train_loader, opt, sched, scaler)
        vl, va, _, _ = evaluate(model, val_loader)
        history.append(dict(epoch=ep, train_loss=tl, val_loss=vl, val_acc=va))
        elapsed = (time.time() - t0) / 60
        print(f"epoch {ep}/{epochs} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.4f} | {elapsed:.1f} menit")
        if va > best:
            best = va; torch.save(model.state_dict(), ckpt_path)
        if max_minutes and elapsed > max_minutes:
            print("Batas waktu tercapai, berhenti."); break
    return history


@torch.no_grad()
def evaluate(model, loader):
    """Mengevaluasi model pada DataLoader berlabel.

    Returns:
        Tuple ``(loss, akurasi, y_true, y_pred)``.
    """
    model.eval()
    loss, ys, ps = 0.0, [], []
    for x, y in tqdm(loader, leave=False, desc="eval"):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(pixel_values=x.to(DEVICE)).logits.float().cpu()
        loss += F.cross_entropy(logits, y, reduction="sum").item()
        ys.append(y); ps.append(logits.argmax(-1))
    yt, yp = torch.cat(ys), torch.cat(ps)
    return loss / len(yt), (yt == yp).float().mean().item(), yt.numpy(), yp.numpy()


def main():
    """Entry point CLI: training end-to-end dan menyimpan ``best.pt`` serta ``history.csv``."""
    seed_everything(SEED)
    print("Device:", DEVICE, "| Data:", DATA_DIR)
    train_loader, val_loader, _ = build_loaders()
    model = build_model().to(DEVICE)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    history = fit(model, train_loader, val_loader, EPOCHS, LR, CKPT_DIR / "best.pt", MAX_MINUTES)
    pd.DataFrame(history).to_csv(WORK / "history.csv", index=False)


if __name__ == "__main__":
    main()
