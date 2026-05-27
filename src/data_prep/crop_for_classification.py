"""
crop_for_classification.py - Cat anh tu BBox de tao dataset Classification (Stage 2)
Nhom 2 - Waste Detection (2-Stage Pipeline)

Doc anh + label YOLO (5 classes) tu data/processed/,
cat tung vung BBox ra thanh anh rieng, luu theo cau truc ImageFolder:

    data/classification/
    ├── train/
    │   ├── Glass/
    │   ├── Metal/
    │   ├── Other/
    │   ├── Paper/
    │   └── Plastic/
    ├── val/
    └── test/

Chay SAU: data_cleaning.py -> Training_dataYolo.py -> split_dataset.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]

# Input: dataset YOLO da split (5 classes)
INPUT_DIR = BASE_DIR / "data" / "processed"

# Output: dataset classification (ImageFolder)
OUTPUT_DIR = BASE_DIR / "data" / "classification"

# Kich thuoc anh output cho classifier
CROP_SIZE = 224

# Padding them xung quanh BBox (10% moi chieu) de lay them context
PAD_RATIO = 0.10

# Class mapping (phai khop voi dataset.yaml)
CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Other",
    3: "Paper",
    4: "Plastic",
}

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# =========================
# Oversample config cho class thieu so
# =========================
# Nhan ban anh crop cua cac class it mau de tang so luong
# factor = 0 nghia la khong oversample (chi giu ban goc)
OVERSAMPLE_FACTORS = {
    "Glass": 2,     # Nhan x3 tong (1 goc + 2 ban sao)
    "Metal": 1,     # Nhan x2
    "Other": 0,     # Khong oversample
    "Paper": 1,     # Nhan x2
    "Plastic": 0,   # Khong oversample (da nhieu)
}


def parse_yolo_label(label_path):
    """Doc file label YOLO, tra ve list cac (class_id, cx, cy, w, h) normalized."""
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            labels.append((cls, cx, cy, w, h))

    return labels


def crop_bbox_with_padding(img, cx, cy, bw, bh, pad_ratio=0.10):
    """
    Cat vung BBox tu anh, co them padding xung quanh.

    Parameters
    ----------
    img : numpy array (H, W, C)
    cx, cy, bw, bh : float (normalized 0-1)
    pad_ratio : float - ty le padding them moi chieu

    Returns
    -------
    crop : numpy array hoac None neu bbox khong hop le
    """
    img_h, img_w = img.shape[:2]

    # Chuyen tu normalized ve pixel
    x_center = cx * img_w
    y_center = cy * img_h
    box_w = bw * img_w
    box_h = bh * img_h

    # Tinh padding (% cua kich thuoc bbox)
    pad_x = box_w * pad_ratio
    pad_y = box_h * pad_ratio

    # Tinh toa do goc tren-trai va goc duoi-phai (co padding)
    x1 = int(max(0, x_center - box_w / 2 - pad_x))
    y1 = int(max(0, y_center - box_h / 2 - pad_y))
    x2 = int(min(img_w, x_center + box_w / 2 + pad_x))
    y2 = int(min(img_h, y_center + box_h / 2 + pad_y))

    # Kiem tra kich thuoc hop le
    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    # Kiem tra crop khong bi rong
    if crop.size == 0:
        return None

    return crop


def apply_augmentation(crop, aug_type):
    """Ap dung augmentation nhe cho ban copy oversample."""
    if aug_type == 0:
        # Lat ngang
        return cv2.flip(crop, 1)
    elif aug_type == 1:
        # Thay doi do sang
        alpha = np.random.uniform(0.7, 1.3)
        beta = np.random.randint(-25, 25)
        return cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    elif aug_type == 2:
        # Lat doc
        return cv2.flip(crop, 0)
    else:
        return crop


def process_split(split):
    """Xu ly 1 split (train/val/test): doc anh + label, crop, luu."""
    img_dir = INPUT_DIR / "images" / split
    lbl_dir = INPUT_DIR / "labels" / split

    if not img_dir.exists():
        print(f"[WARN] Khong tim thay: {img_dir}")
        return {}

    # Tim tat ca anh trong split
    img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"\n[{split.upper()}] Tim thay {len(img_paths)} anh")

    crop_counter = Counter()
    total_skipped = 0

    for img_path in img_paths:
        # Tim file label tuong ung
        rel = img_path.relative_to(img_dir)
        lbl_path = (lbl_dir / rel).with_suffix(".txt")

        # Doc labels
        labels = parse_yolo_label(lbl_path)
        if not labels:
            continue

        # Doc anh
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Khong doc duoc: {img_path.name}")
            continue

        # Crop tung BBox
        for idx, (cls_id, cx, cy, bw, bh) in enumerate(labels):
            if cls_id not in CLASS_NAMES:
                total_skipped += 1
                continue

            class_name = CLASS_NAMES[cls_id]

            # Crop voi padding
            crop = crop_bbox_with_padding(img, cx, cy, bw, bh, PAD_RATIO)
            if crop is None:
                total_skipped += 1
                continue

            # Resize ve CROP_SIZE x CROP_SIZE
            crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                       interpolation=cv2.INTER_LINEAR)

            # Tao thu muc output
            out_dir = OUTPUT_DIR / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Luu crop goc
            out_name = f"{img_path.stem}_bbox{idx}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop_resized)
            crop_counter[class_name] += 1

            # Oversample: tao them ban sao voi augmentation nhe
            # Chi ap dung cho tap train
            if split == "train":
                factor = OVERSAMPLE_FACTORS.get(class_name, 0)
                for aug_idx in range(factor):
                    aug_crop = apply_augmentation(crop_resized, aug_idx)
                    aug_name = f"{img_path.stem}_bbox{idx}_aug{aug_idx}.jpg"
                    aug_path = out_dir / aug_name
                    cv2.imwrite(str(aug_path), aug_crop)
                    crop_counter[class_name] += 1

    if total_skipped > 0:
        print(f"  [INFO] Bo qua {total_skipped} bbox khong hop le")

    return crop_counter


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("  CROP FOR CLASSIFICATION - Stage 2 Dataset")
    print("=" * 60)
    print(f"  Input  : {INPUT_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Crop size : {CROP_SIZE}x{CROP_SIZE}")
    print(f"  Padding   : {PAD_RATIO*100:.0f}%")
    print(f"  Oversample (train only): {OVERSAMPLE_FACTORS}")
    print("=" * 60)

    all_stats = {}
    grand_total = 0

    for split in SPLITS:
        stats = process_split(split)
        all_stats[split] = stats

        total = sum(stats.values())
        grand_total += total

        print(f"\n  [{split.upper()}] Tong: {total} crops")
        for cls_name in sorted(CLASS_NAMES.values()):
            cnt = stats.get(cls_name, 0)
            bar = "█" * (cnt // 20)
            print(f"    {cls_name:10s}: {cnt:5d}  {bar}")

    print(f"\n{'=' * 60}")
    print(f"  TONG CONG: {grand_total} crops")
    print(f"  Luu tai: {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    # Kiem tra can bang
    if "train" in all_stats and all_stats["train"]:
        train_stats = all_stats["train"]
        max_cnt = max(train_stats.values())
        min_cnt = min(train_stats.values())
        ratio = max_cnt / max(min_cnt, 1)
        print(f"\n  [INFO] Ty le max/min trong tap train: {ratio:.1f}:1")
        if ratio > 5:
            print("  [WARN] Van con mat can bang! Can tang oversample cho class thieu so.")
        else:
            print("  [OK] Muc can bang chap nhan duoc.")

    print("\n[HOAN TAT] Crop xong! Dataset san sang cho Stage 2.")
