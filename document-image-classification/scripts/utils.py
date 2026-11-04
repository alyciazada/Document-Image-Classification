"""Fungsi bantu: seed dan visualisasi confusion matrix."""
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def seed_everything(seed):
    """Set seed Python, NumPy, dan PyTorch agar hasil reproducible."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(y_true, y_pred, class_names, path=None):
    """Menggambar confusion matrix ternormalisasi per baris.

    Args:
        y_true: Label sebenarnya.
        y_pred: Label prediksi.
        class_names: Nama kelas.
        path: Jika diberikan, gambar disimpan ke path ini.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=False)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya"); fig.tight_layout()
    if path: fig.savefig(path, dpi=150)
    plt.show()
