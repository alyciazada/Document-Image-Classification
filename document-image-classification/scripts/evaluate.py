"""Evaluasi pada data validasi. Jalankan: ``python -m src.evaluate``."""
import matplotlib

matplotlib.use("Agg")

import torch
from sklearn.metrics import classification_report

from .config import CKPT_DIR, CLASS_NAMES, DEVICE, SEED, WORK
from .data import build_loaders
from .model import build_model
from .train import evaluate
from .utils import plot_confusion_matrix, seed_everything


def main():
    """Entry point CLI: memuat ``best.pt``, mencetak laporan per kelas, dan menyimpan confusion matrix."""
    seed_everything(SEED)
    model = build_model()
    model.load_state_dict(torch.load(CKPT_DIR / "best.pt", map_location="cpu"))
    model.to(DEVICE)
    _, val_loader, _ = build_loaders()
    vl, va, y_true, y_pred = evaluate(model, val_loader)
    print(f"val_loss={vl:.4f} val_acc={va:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))
    WORK.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES, WORK / "confusion_matrix.png")


if __name__ == "__main__":
    main()
