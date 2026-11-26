import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_DIR = "runs/exp1"
OUT_DIR = Path("assets")
DATASET = "nkirschi/oxford-flowers"


def load_model(device):
    classes = json.load(open(f"{RUN_DIR}/classes.json", encoding="utf-8"))
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(f"{RUN_DIR}/best.pt", map_location=device))
    return model.to(device).eval(), classes


def main():
    OUT_DIR.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(device)

    tf = transforms.Compose(
        [
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_raw = load_dataset(DATASET, split="test")  # 原圖，之後顯示錯誤案例用

    def fn(batch):
        batch["pixel_values"] = [tf(img.convert("RGB")) for img in batch["image"]]
        return batch

    test_tf = test_raw.with_transform(fn)

    def collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["label"] for b in batch])
        return x, y

    loader = DataLoader(test_tf, batch_size=64, num_workers=0, collate_fn=collate)

    # ---------- 推論整個 test 集 ----------
    preds, confs, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            preds += pred.cpu().tolist()
            confs += conf.cpu().tolist()
            labels += y.tolist()

    preds, confs, labels = np.array(preds), np.array(confs), np.array(labels)
    wrong = np.where(preds != labels)[0]
    acc = (preds == labels).mean()
    print(f"Test 準確率: {acc:.3%}（{len(labels) - len(wrong)}/{len(labels)}）")
    print(f"答對時平均信心 {confs[preds == labels].mean():.1%} | "
          f"答錯時平均信心 {confs[wrong].mean():.1%}")

    # ---------- 混淆矩陣 ----------
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))

    # 最常搞混的組合（排除對角線）
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(off[i, j], i, j) for i in range(len(classes)) for j in range(len(classes)) if off[i, j]]
    pairs.sort(reverse=True)
    print("\n最常搞混的組合（真實 -> 預測）:")
    for n, i, j in pairs[:10]:
        print(f"  花卉編號 {classes[i]} -> {classes[j]}：{n} 次")

    # 每個類別的準確率，找出最弱的
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    print("\n準確率最低的 5 個類別:")
    for i in np.argsort(per_class)[:5]:
        print(f"  花卉編號 {classes[i]}：{per_class[i]:.0%}（共 {cm[i].sum()} 張）")

    # 把對角線（答對）去掉，只畫錯誤，才看得出問題在哪
    plt.figure(figsize=(8, 7))
    plt.imshow(off, cmap="Reds")
    plt.colorbar(label="Error count")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.title("Confusion matrix (errors only)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # ---------- 錯誤案例 ----------
    show = wrong[np.argsort(-confs[wrong])][:12]  # 優先看「很有信心卻答錯」的
    if len(show):
        fig, axes = plt.subplots(3, 4, figsize=(12, 9))
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, show):
            ax.imshow(test_raw[int(idx)]["image"])
            ax.set_title(
                f"true {classes[labels[idx]]} / pred {classes[preds[idx]]}\nconf {confs[idx]:.0%}",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(OUT_DIR / "errors.png", dpi=120)
        plt.close()

    print(f"\n已輸出：{OUT_DIR / 'confusion_matrix.png'}、{OUT_DIR / 'errors.png'}")


if __name__ == "__main__":
    main()