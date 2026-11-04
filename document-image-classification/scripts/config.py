"""Konfigurasi global proyek: lokasi data, kelas, dan hyperparameter."""
import glob
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
COMP = "document-image-classification-sd234504"
ON_KAGGLE = Path("/kaggle/working").exists()


def find_data_dir():
    """Mencari folder dataset yang berisi ``train.csv``.

    Urutan pencarian: lokasi data kompetisi di Kaggle, lalu folder ``data/`` di repo,
    lalu pencarian ``train.csv`` di ``/kaggle/input``.

    Returns:
        Path folder dataset. Jika tidak ditemukan, mengembalikan ``ROOT/data``.
    """
    candidates = [Path(f"/kaggle/input/{COMP}"), Path(f"/kaggle/input/competitions/{COMP}"), ROOT / "data"]
    for p in candidates:
        if (p / "train.csv").exists():
            return p
    hits = [h for pat in ("*/train.csv", "*/*/train.csv", "*/*/*/train.csv") for h in glob.glob("/kaggle/input/" + pat)]
    return Path(hits[0]).parent if hits else ROOT / "data"


DATA_DIR = find_data_dir()
WORK = Path("/kaggle/working") if ON_KAGGLE else ROOT / "outputs"      # cache, history, submission
CKPT_DIR = WORK if ON_KAGGLE else ROOT / "checkpoint"                   # bobot model (best.pt)

CLASS_NAMES = ["advertisement", "budget", "email", "file_folder", "form", "handwritten",
               "invoice", "letter", "memo", "news_article", "presentation", "questionnaire",
               "resume", "scientific_publication", "scientific_report", "specification"]
NUM_CLASSES = len(CLASS_NAMES)

SIZE = 224
MODEL_NAME = "microsoft/dit-base"
BATCH_SIZE, VAL_FRACTION = 32, 0.1
NUM_WORKERS = 0 if os.name == "nt" else 2
SEED = 42

EPOCHS, LR, WEIGHT_DECAY, LABEL_SMOOTHING = 3, 3e-5, 0.05, 0.1
MAX_MINUTES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
