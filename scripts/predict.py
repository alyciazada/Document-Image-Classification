"""Prediksi test set dan pembuatan submission. Jalankan: ``python -m src.predict``."""
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import CKPT_DIR, DATA_DIR, DEVICE, WORK
from .data import build_test_loader
from .model import build_model


@torch.no_grad()
def predict_probs(model, loader):
    """Menghitung probabilitas softmax untuk seluruh DataLoader.

    Returns:
        Array (N, 16).
    """
    model.eval(); out = []
    for x in tqdm(loader, desc="test"):
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            out.append(model(pixel_values=x.to(DEVICE)).logits.float().softmax(-1).cpu())
    return torch.cat(out).numpy()


def main():
    """Entry point CLI: memuat ``best.pt``, memprediksi test, dan menulis ``submission.csv``."""
    model = build_model()
    model.load_state_dict(torch.load(CKPT_DIR / "best.pt", map_location="cpu"))
    model.to(DEVICE)
    test_loader, test_df = build_test_loader()
    probs = predict_probs(model, test_loader)
    WORK.mkdir(parents=True, exist_ok=True)
    np.save(WORK / "test_probs.npy", probs)
    sub = pd.DataFrame({"id": test_df["id"], "label": probs.argmax(1)})
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(sub.columns) == list(sample.columns) and len(sub) == len(sample) and (sub.id == sample.id).all()
    sub.to_csv(WORK / "submission.csv", index=False)
    print("Tersimpan:", WORK / "submission.csv", sub.shape)


if __name__ == "__main__":
    main()
