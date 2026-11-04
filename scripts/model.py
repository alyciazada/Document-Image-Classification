"""Definisi model: DiT-base pre-trained dengan head 16 kelas."""
from .config import CLASS_NAMES, MODEL_NAME, NUM_CLASSES


def build_model(name=MODEL_NAME):
    """Membuat model Transformer pre-trained dengan head baru 16 kelas.

    Args:
        name: Nama model di HuggingFace Hub.

    Returns:
        Model PyTorch (keluaran logits (B, 16)).
    """
    from transformers import AutoModelForImageClassification
    return AutoModelForImageClassification.from_pretrained(
        name, num_labels=NUM_CLASSES, id2label=dict(enumerate(CLASS_NAMES)),
        label2id={c: i for i, c in enumerate(CLASS_NAMES)}, ignore_mismatched_sizes=True)
