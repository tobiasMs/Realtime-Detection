# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 16:12:17 2025

@author: tobias.sulistiyo
"""

# train_ddd.py

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report
import numpy as np

# ─── DataLoader ─────────────────────────────────────────────────────────────
def get_dataloaders(data_root="data/DDD", img_size=224, batch_size=32, workers=4):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)), transforms.CenterCrop(img_size),
        transforms.ToTensor(), transforms.Normalize(mean,std)
    ])

    sets = {}
    loaders = {}
    for split in ("train","val","test"):
        sets[split] = datasets.ImageFolder(os.path.join(data_root, split),
                                           transform=train_tf if split=="train" else val_tf)
        loaders[split] = torch.utils.data.DataLoader(
            sets[split], batch_size=batch_size, shuffle=(split=="train"),
            num_workers=workers, pin_memory=True
        )
    return sets, loaders

# ─── Build Model ────────────────────────────────────────────────────────────
def build_model(num_classes=2):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

# ─── Training Loop ──────────────────────────────────────────────────────────
def train(data_root="data/DDD", out_dir="runs/mobilenetv2",
          epochs=5, lr=3e-4, bs=32, img_size=224, workers=4, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    sets, loaders = get_dataloaders(data_root, img_size, bs, workers)
    model = build_model(num_classes=2).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    for ep in range(1, epochs+1):
        # train
        model.train()
        correct, seen, losses = 0, 0, []
        for imgs, labels in loaders["train"]:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            losses.append(loss.item())
            correct += (logits.argmax(1)==labels).sum().item()
            seen += imgs.size(0)
        train_acc = correct/seen

        # validate
        model.eval()
        y_true, y_pred = [], []
        with torch.inference_mode():
            for imgs, labels in loaders["val"]:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(logits.argmax(1).cpu().numpy())
        val_acc = np.mean(np.array(y_true)==np.array(y_pred))

        print(f"[Epoch {ep}] Loss={np.mean(losses):.4f} TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f}")
        print(classification_report(y_true, y_pred, target_names=["non_drowsy","drowsy"]))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict()}, os.path.join(out_dir,"best.pt"))
            print(">>> Best model saved!")

        scheduler.step()

    # test evaluation
    ckpt = torch.load(os.path.join(out_dir,"best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for imgs, labels in loaders["test"]:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())
    print("\n=== TEST METRICS ===")
    print(classification_report(y_true, y_pred, target_names=["non_drowsy","drowsy"]))

if __name__=="__main__":
    train(epochs=10)
