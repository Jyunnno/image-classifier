import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 1. 設定 ----------
CFG = dict(
    dataset="nkirschi/oxford-flowers",
    out_dir="runs/exp1",
    img_size=224,
    batch_size=32,
    epochs=12,
    freeze_epochs=2,       # 前幾個 epoch 只訓練輸出層
    lr_head=1e-3,
    lr_finetune=1e-4,
    weight_decay=1e-4,
    val_ratio=0.15,
    num_workers=0,         # Windows 先用 0，避免卡住
    seed=42,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------- 2. 資料 ----------
def build_loaders(cfg):
    size = cfg["img_size"]
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(), norm,
    ])

    ds = load_dataset(cfg["dataset"])
    split = ds["train"].train_test_split(test_size=cfg["val_ratio"], seed=cfg["seed"])
    train_ds, val_ds, test_ds = split["train"], split["test"], ds["test"]
    classes = ds["train"].features["label"].names

    def make_transform(tf):
        def fn(batch):
            batch["pixel_values"] = [tf(img.convert("RGB")) for img in batch["image"]]
            return batch
        return fn

    train_ds.set_transform(make_transform(tf_train))
    val_ds.set_transform(make_transform(tf_eval))
    test_ds.set_transform(make_transform(tf_eval))

    def collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["label"] for b in batch])
        return x, y

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              pin_memory=True, collate_fn=collate)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, **kw),
            DataLoader(test_ds, **kw),
            classes)

# ---------- 3. 模型 ----------
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def set_backbone_trainable(model, trainable):
    for name, p in model.named_parameters():
        if not name.startswith("fc"):
            p.requires_grad = trainable

# ---------- 4. 訓練 / 驗證 ----------
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = device == "cuda"
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n

# ---------- 5. 主流程 ----------
def main():
    cfg = CFG
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("裝置:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl, classes = build_loaders(cfg)
    print(f"類別數: {len(classes)} | train {len(train_dl.dataset)} "
          f"val {len(val_dl.dataset)} test {len(test_dl.dataset)}")
    json.dump(classes, open(out_dir / "classes.json", "w", encoding="utf-8"), ensure_ascii=False)

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr_head"], weight_decay=cfg["weight_decay"])
    scheduler = None

    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])
    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        if epoch == cfg["freeze_epochs"]:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr_finetune"], weight_decay=cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - cfg["freeze_epochs"])
            print(">>> 解凍骨幹，開始微調全部參數")

        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        if scheduler:
            scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
            history[k].append(v)

        flag = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")
            flag = "  <- 最佳，已儲存"

        print(f"[{epoch+1:02d}/{cfg['epochs']}] "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.0f}s{flag}")

    # 學習曲線
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train"); ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_dir / "curves.png", dpi=120)

    # 最後用最佳模型評估 test（只做這一次）
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    te_loss, te_acc = run_epoch(model, test_dl, criterion, device)
    print(f"最佳 val acc {best_acc:.3f} | test acc {te_acc:.3f}")

if __name__ == "__main__":
    main()