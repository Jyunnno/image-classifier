import argparse
import json
import os
import random

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

RUN_DIR = "runs/exp1"


def load_model(device):
    classes = json.load(open(f"{RUN_DIR}/classes.json", encoding="utf-8"))
    model = models.resnet18(weights=None)  # 不用再下載預訓練權重，等一下載入自己的
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(f"{RUN_DIR}/best.pt", map_location=device))
    model.to(device).eval()
    return model, classes


def get_transform(size=224):
    # 必須跟訓練時的驗證/測試預處理完全一致
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def predict(model, img, classes, device, topk=3):
    x = get_transform()(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, topk)
    return [(classes[i], p.item()) for p, i in zip(top.values, top.indices)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="圖片路徑")
    parser.add_argument("--random", action="store_true", help="從 test 集隨機抽一張")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, classes = load_model(device)

    truth = None
    if args.random:
        from datasets import load_dataset

        test = load_dataset("nkirschi/oxford-flowers", split="test")
        item = test[random.randrange(len(test))]
        img, truth = item["image"], classes[item["label"]]
    elif args.image:
        img = Image.open(args.image)
    else:
        parser.error("請提供 --image 圖片路徑，或使用 --random")

    results = predict(model, img, classes, device)
    if truth is not None:
        print(f"正確答案: 花卉編號 {truth}")
    print("預測結果:")
    for rank, (name, p) in enumerate(results, 1):
        print(f"  {rank}. 花卉編號 {name}  信心 {p:.1%}")
    if truth is not None:
        print("答對了" if results[0][0] == truth else "答錯了")


if __name__ == "__main__":
    main()