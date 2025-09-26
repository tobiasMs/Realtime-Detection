# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:05:24 2025

@author: tobias.sulistiyo
"""

import os, shutil, random
from pathlib import Path

def prep_split(src_root=r"C:\Users\tobias.sulistiyo\DrowsyDetection",
               dst_root="data/DDD", seed=42,
               tvt=(0.75, 0.15, 0.10)):
    random.seed(seed)

    # Pastikan sesuai dengan nama folder asli
    cls_map = {
        "Drowsy": "drowsy",
        "Non Drowsy": "non_drowsy"   # ubah kalau aslinya Non_Drowsy
    }

    src_root = Path(src_root)
    dst_root = Path(dst_root)

    # Buat folder tujuan
    for split in ("train", "val", "test"):
        for c in cls_map.values():
            (dst_root / split / c).mkdir(parents=True, exist_ok=True)

    for src_cls, dst_cls in cls_map.items():
        src_path = src_root / src_cls
        imgs = [p for p in src_path.rglob("*.*")
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

        print(f"[CHECK] {src_cls} → {len(imgs)} file ditemukan")

        if not imgs:
            print(f"[WARNING] Tidak ada file di {src_path}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * tvt[0])
        n_val = int(n * tvt[1])
        n_test = n - n_train - n_val

        splits = [
            ("train", imgs[:n_train]),
            ("val", imgs[n_train:n_train + n_val]),
            ("test", imgs[n_train + n_val:])
        ]

        for split, items in splits:
            for i, p in enumerate(items):
                dst = dst_root / split / dst_cls / f"{dst_cls}_{i:07d}{p.suffix.lower()}"
                shutil.copy2(p, dst)

        print(f"[DONE] {src_cls}: train={n_train}, val={n_val}, test={n_test}")

    print("✅ Dataset split selesai!")

if __name__ == "__main__":
    prep_split()


