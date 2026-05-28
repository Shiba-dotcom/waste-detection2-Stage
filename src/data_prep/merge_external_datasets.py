"""
merge_external_datasets.py - Gop dataset ben ngoai vao dataset Classification (Stage 2)
Nhom 2 - Waste Detection (2-Stage Pipeline)

Gop anh tu TrashNet va RealWaste vao dataset TACO crops da co,
tao dataset lon hon va can bang hon cho viec train classifier.

Su dung tren Kaggle:
  - TrashNet:  /kaggle/input/trashnet/dataset-resized/
  - RealWaste: /kaggle/input/realwaste-image-classification/RealWaste/

Hoac chay local neu da download dataset ve.

Chay SAU: crop_for_classification.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import shutil
import random
from pathlib import Path
from collections import Counter

# =========================
# Config
# =========================
random.seed(42)

# --- Phat hien moi truong ---
# Chú ý: Ép buộc bằng False vì chúng ta đã chép toàn bộ data vào Working Dir trên Kaggle,
# do đó cấu trúc thư mục giờ đây y hệt như Local.
# Tự động phát hiện môi trường
import os as _os
ON_KAGGLE = _os.environ.get("ON_KAGGLE", "0") == "1" or _os.path.exists("/kaggle/working")

if ON_KAGGLE:
    # Tren Kaggle: TACO crops duợc tạo ra bởi crop_for_classification.py
    TACO_CROPS_DIR = Path("/kaggle/working/waste-detection2-Stage/data/classification")
    # Dataset ngoài
    TRASHNET_DIR = Path("/kaggle/working/waste-detection2-Stage/data/external/TrashNet")
    REALWASTE_DIR = Path("/kaggle/working/waste-detection2-Stage/data/external/RealWaste")
    # Output: gop tat ca vao day
    OUTPUT_DIR = Path("/kaggle/working/waste-detection2-Stage/data/classification_merged")
else:
    # Chay local
    BASE_DIR = Path(__file__).resolve().parents[2]
    TACO_CROPS_DIR = BASE_DIR / "data" / "classification"
    
    # [Đã cập nhật để dễ copy paste]:
    # Thư mục gốc chứa các dataset ngoài
    EXTERNAL_DIR = BASE_DIR / "data" / "external"
    
    # Bạn chỉ cần đổi tên thư mục giải nén thành "TrashNet" và "RealWaste" 
    # rồi dán vào bên trong thư mục "data/external"
    TRASHNET_DIR = EXTERNAL_DIR / "TrashNet"
    REALWASTE_DIR = EXTERNAL_DIR / "RealWaste"
    
    OUTPUT_DIR = BASE_DIR / "data" / "classification_merged"

# --- Gioi han so luong toi da moi class (cap) ---
# Plastic da nhieu, can gioi han de tranh mat can bang
MAX_PER_CLASS = 2500

# --- Ty le chia cho dataset ngoai (chi ap dung khi khong co san val/test) ---
EXTERNAL_VAL_RATIO = 0.15
EXTERNAL_TEST_RATIO = 0.15

# --- Mapping tu ten class cua tung dataset sang 5 class cua ta ---
# TrashNet classes: cardboard, glass, metal, paper, plastic, trash
TRASHNET_MAPPING = {
    "cardboard": "Paper",      # Cardboard la dang giay bìa
    "glass":     "Glass",
    "metal":     "Metal",
    "paper":     "Paper",
    "plastic":   "Plastic",
    "trash":     "Other",
}

# RealWaste classes: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash,
#                    Paper, Plastic, Textile Trash, Vegetation
REALWASTE_MAPPING = {
    "Cardboard":           "Paper",
    "Food Organics":       "Other",
    "Glass":               "Glass",
    "Metal":               "Metal",
    "Miscellaneous Trash": "Other",
    "Paper":               "Paper",
    "Plastic":             "Plastic",
    "Textile Trash":       "Other",
    "Vegetation":          "Other",
}

OUR_CLASSES = ["Glass", "Metal", "Other", "Paper", "Plastic"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# =========================
# Helper Functions
# =========================
def copy_images(src_paths, dst_dir, prefix=""):
    """Copy danh sach anh vao thu muc dich, them prefix de tranh trung ten."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in src_paths:
        if prefix:
            dst_name = f"{prefix}_{src.name}"
        else:
            dst_name = src.name
        dst = dst_dir / dst_name
        # Tranh ghi de neu da ton tai
        if dst.exists():
            dst_name = f"{prefix}_{count}_{src.name}"
            dst = dst_dir / dst_name
        shutil.copy2(str(src), str(dst))
        count += 1
    return count


def collect_images_from_dir(root_dir, class_mapping):
    """Thu thap anh tu cau truc ImageFolder, map class name."""
    result = {cls: [] for cls in OUR_CLASSES}

    if not root_dir.exists():
        print(f"  [WARN] Khong tim thay: {root_dir}")
        return result

    for class_dir in root_dir.iterdir():
        if not class_dir.is_dir():
            continue

        src_class = class_dir.name
        mapped_class = class_mapping.get(src_class)

        if mapped_class is None:
            print(f"  [SKIP] Class '{src_class}' khong co trong mapping")
            continue

        imgs = [p for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS]
        result[mapped_class].extend(imgs)

    return result


def split_list(items, val_ratio, test_ratio):
    """Chia list thanh train/val/test."""
    random.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]


# =========================
# Main
# =========================
if __name__ == "__main__":
    print("=" * 65)
    print("  MERGE EXTERNAL DATASETS - Classification Stage 2")
    print("=" * 65)
    print(f"  Moi truong  : {'Kaggle' if ON_KAGGLE else 'Local'}")
    print(f"  TACO crops  : {TACO_CROPS_DIR}")
    print(f"  TrashNet    : {TRASHNET_DIR}")
    print(f"  RealWaste   : {REALWASTE_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  Max/class   : {MAX_PER_CLASS}")
    print("=" * 65)

    # ── Buoc 1: Thu thap anh TACO crops (da co san train/val/test) ──
    print("\n[Buoc 1] Thu thap TACO crops...")
    taco_by_split = {}
    for split in ["train", "val", "test"]:
        split_dir = TACO_CROPS_DIR / split
        if split_dir.exists():
            taco_by_split[split] = {cls: [] for cls in OUR_CLASSES}
            for cls in OUR_CLASSES:
                cls_dir = split_dir / cls
                if cls_dir.exists():
                    imgs = [p for p in cls_dir.iterdir()
                            if p.is_file() and p.suffix.lower() in IMG_EXTS]
                    taco_by_split[split][cls] = imgs
            total = sum(len(v) for v in taco_by_split[split].values())
            print(f"  [{split}] {total} anh")
        else:
            print(f"  [WARN] Khong tim thay: {split_dir}")

    # ── Buoc 2: Thu thap TrashNet ──
    print("\n[Buoc 2] Thu thap TrashNet...")
    trashnet_imgs = collect_images_from_dir(TRASHNET_DIR, TRASHNET_MAPPING)
    for cls in OUR_CLASSES:
        print(f"  {cls:10s}: {len(trashnet_imgs[cls]):5d}")

    # ── Buoc 3: Thu thap RealWaste ──
    print("\n[Buoc 3] Thu thap RealWaste...")
    realwaste_imgs = collect_images_from_dir(REALWASTE_DIR, REALWASTE_MAPPING)
    for cls in OUR_CLASSES:
        print(f"  {cls:10s}: {len(realwaste_imgs[cls]):5d}")

    # ── Buoc 4: Chia dataset ngoai thanh train/val/test ──
    print("\n[Buoc 4] Chia dataset ngoai thanh train/val/test...")
    external_by_split = {s: {c: [] for c in OUR_CLASSES} for s in ["train", "val", "test"]}

    for cls in OUR_CLASSES:
        # Gop TrashNet + RealWaste
        all_external = trashnet_imgs[cls] + realwaste_imgs[cls]

        if not all_external:
            continue

        train, val, test = split_list(all_external, EXTERNAL_VAL_RATIO, EXTERNAL_TEST_RATIO)
        external_by_split["train"][cls] = train
        external_by_split["val"][cls] = val
        external_by_split["test"][cls] = test

        print(f"  {cls:10s}: train={len(train):4d} val={len(val):4d} test={len(test):4d}")

    # ── Buoc 5: Gop tat ca va ap dung cap ──
    print("\n[Buoc 5] Gop va copy anh...")

    final_stats = {}

    for split in ["train", "val", "test"]:
        print(f"\n  --- {split.upper()} ---")
        final_stats[split] = {}

        for cls in OUR_CLASSES:
            # Gop TACO + external
            taco_imgs = taco_by_split.get(split, {}).get(cls, [])
            ext_imgs = external_by_split[split][cls]

            # Tao output dir
            out_dir = OUTPUT_DIR / split / cls

            # Copy TACO crops
            n_taco = copy_images(taco_imgs, out_dir, prefix="taco")

            # Copy external
            n_ext = copy_images(ext_imgs, out_dir, prefix="ext")

            total = n_taco + n_ext

            # Ap dung cap chi cho tap train
            if split == "train" and total > MAX_PER_CLASS:
                # Lay random MAX_PER_CLASS anh, xoa phan du
                all_files = list(out_dir.iterdir())
                random.shuffle(all_files)
                to_remove = all_files[MAX_PER_CLASS:]
                for f in to_remove:
                    f.unlink()
                total = MAX_PER_CLASS
                print(f"    {cls:10s}: {n_taco} (TACO) + {n_ext} (ext) "
                      f"= {n_taco+n_ext} -> CAP {MAX_PER_CLASS}")
            else:
                print(f"    {cls:10s}: {n_taco} (TACO) + {n_ext} (ext) = {total}")

            final_stats[split][cls] = total

    # ── Buoc 6: Thong ke cuoi cung ──
    print(f"\n{'=' * 65}")
    print("  THONG KE CUOI CUNG")
    print(f"{'=' * 65}")

    for split in ["train", "val", "test"]:
        total = sum(final_stats[split].values())
        print(f"\n  [{split.upper()}] Tong: {total}")
        for cls in OUR_CLASSES:
            cnt = final_stats[split].get(cls, 0)
            bar = "█" * (cnt // 30)
            print(f"    {cls:10s}: {cnt:5d}  {bar}")

    # Kiem tra can bang train
    train_counts = list(final_stats["train"].values())
    if train_counts:
        ratio = max(train_counts) / max(min(train_counts), 1)
        print(f"\n  Ty le max/min (train): {ratio:.1f}:1")
        if ratio <= 3:
            print("  [OK] Can bang tot!")
        else:
            print("  [WARN] Con mat can bang, can dieu chinh MAX_PER_CLASS hoac oversample")

    print(f"\n  Dataset merged luu tai: {OUTPUT_DIR}")
    print(f"{'=' * 65}")
    print("[HOAN TAT] Merge xong!")
