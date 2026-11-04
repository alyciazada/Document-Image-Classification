"""Pembacaan gambar, cache, Dataset, dan DataLoader."""
import glob
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import BATCH_SIZE, DATA_DIR, NUM_WORKERS, SEED, SIZE, VAL_FRACTION, WORK


def resolve_path(root, filename):
    """Mencari lokasi file gambar di disk.

    CSV menulis ``train/invoice/x.tif`` sedangkan di disk bisa ``train/train/invoice/x.tif``.

    Args:
        root: Folder dataset.
        filename: Kolom ``filename`` dari CSV.

    Returns:
        Path file yang ditemukan.
    """
    root = Path(root)
    for p in (root / filename, root / filename.split("/")[0] / filename):
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} tidak ditemukan di {root}")


def load_image(path):
    """Membuka gambar dengan fallback decoder; file korup diganti halaman putih.

    Args:
        path: Lokasi file gambar.

    Returns:
        Citra PIL mode RGB.
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        pass
    try:
        import tifffile
        return Image.fromarray(tifffile.imread(str(path))).convert("RGB")
    except Exception:
        pass
    warnings.warn(f"Gambar tidak terbaca, diganti halaman putih: {path}")
    return Image.new("RGB", (224, 224), (255, 255, 255))


def prep_image(path, size=SIZE):
    """Decode satu gambar -> grayscale uint8 ``size`` x ``size``."""
    img = load_image(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def find_existing(name):
    """Mencari file hasil run sebelumnya (di /kaggle/working atau di input notebook lain).

    Returns:
        Path file jika ada, selain itu None.
    """
    for pattern in (str(WORK / name), f"/kaggle/input/*/{name}", f"/kaggle/input/*/*/{name}", f"/kaggle/input/*/*/*/{name}"):
        hits = glob.glob(pattern)
        if hits:
            return Path(hits[0])
    return None


def build_cache(df, name, workers=4):
    """Membuat (atau memakai ulang) cache array gambar uint8 (N, 224, 224).

    Args:
        df: DataFrame dengan kolom ``filename``.
        name: Nama file cache ``.npy``.
        workers: Jumlah thread decoding.

    Returns:
        Path file cache.
    """
    found = find_existing(name)
    if found is not None:
        print("Memakai cache:", found)
        return found
    path = WORK / name
    paths = [resolve_path(DATA_DIR, f) for f in df["filename"]]
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(paths), SIZE, SIZE))
    with ThreadPoolExecutor(workers) as ex:
        for i, a in enumerate(tqdm(ex.map(prep_image, paths), total=len(paths), desc=name)):
            arr[i] = a
    arr.flush()
    return path


class CachedDataset(Dataset):
    """Dataset dari cache: normalisasi (mean 0.5, std 0.5), 3 channel, augmentasi geser ringan.

    Args:
        cache_path: File ``.npy`` hasil ``build_cache``.
        labels: Array label (None untuk test).
        train: Jika True, aktifkan augmentasi geser acak kecil.
    """

    def __init__(self, cache_path, labels=None, train=False):
        """Memuat cache (memory-mapped) dan menyimpan label."""
        self.arr = np.load(cache_path, mmap_mode="r")
        self.labels, self.train = labels, train

    def __len__(self):
        """Jumlah sampel."""
        return len(self.arr)

    def __getitem__(self, i):
        """Mengambil satu sampel: ``(tensor, label)`` atau hanya tensor bila tanpa label."""
        x = torch.from_numpy(np.array(self.arr[i])).float().div_(255.0)
        if self.train:
            dx, dy = np.random.randint(-8, 9, size=2)
            x = torch.roll(x, shifts=(int(dy), int(dx)), dims=(0, 1))
        x = ((x - 0.5) / 0.5).unsqueeze(0).expand(3, -1, -1).contiguous()
        return x if self.labels is None else (x, torch.tensor(int(self.labels[i])))


def build_loaders():
    """Membuat DataLoader train & validasi (stratified split) dari train.csv.

    Returns:
        Tuple ``(train_loader, val_loader, train_df)``.
    """
    df = pd.read_csv(DATA_DIR / "train.csv")
    cache = build_cache(df, f"cache_train_{SIZE}.npy")
    tr_i, va_i = train_test_split(np.arange(len(df)), test_size=VAL_FRACTION,
                                  stratify=df["label"], random_state=SEED)
    y = df["label"].values
    train = Subset(CachedDataset(cache, y, train=True), tr_i)
    val = Subset(CachedDataset(cache, y), va_i)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=True)
    return (DataLoader(train, BATCH_SIZE, shuffle=True, drop_last=True, **kw),
            DataLoader(val, BATCH_SIZE * 2, **kw), df)


def build_test_loader():
    """Membuat DataLoader untuk test.csv.

    Returns:
        Tuple ``(test_loader, test_df)``.
    """
    df = pd.read_csv(DATA_DIR / "test.csv")
    cache = build_cache(df, f"cache_test_{SIZE}.npy")
    return DataLoader(CachedDataset(cache), BATCH_SIZE * 2, num_workers=NUM_WORKERS, pin_memory=True), df
