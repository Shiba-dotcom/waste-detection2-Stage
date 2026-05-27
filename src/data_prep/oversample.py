import cv2
import numpy as np
import random
from pathlib import Path
import os

# =========================
# config
# =========================
base = Path(__file__).resolve().parents[2] / "data" / "processed"

img_dir = base / "images" / "train"
label_dir = base / "labels" / "train"

GLASS_ID = 4
PAPER_ID = 3
METAL_ID = 2

glass_factor = 2
paper_factor = 1    
metal_factor = 1

# =========================
# Data Augmentation Functions
# =========================
def apply_augmentation(img, labels, aug_type):
    """
    Áp dụng phép biến đổi ảnh và cập nhật tọa độ bounding box nếu cần.
    """
    aug_labels = labels.copy()
    
    if aug_type == "flip":
        # Lật ngang (Horizontal Flip)
        img = cv2.flip(img, 1)
        aug_labels = []
        for line in labels:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = parts
                # Lật x_center qua trục tung
                new_cx = 1.0 - float(cx)
                aug_labels.append(f"{cls} {new_cx:.6f} {cy} {w} {h}")
                
    elif aug_type == "color":
        # Biến đổi độ sáng (Brightness) và độ tương phản (Contrast)
        alpha = random.uniform(0.7, 1.3) # Tương phản [0.7, 1.3]
        beta = random.randint(-30, 30)   # Sáng tối [-30, 30]
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    elif aug_type == "noise":
        # Thêm nhiễu Gauss (Gaussian Noise)
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
    elif aug_type == "blur":
        # Làm mờ (Gaussian Blur)
        ksize = random.choice([3, 5, 7])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
    return img, aug_labels

# =========================
# helper
# =========================
def contains_class(label_path, class_id):
    if not label_path.exists():
        return False
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) > 0 and int(parts[0]) == class_id:
                return True
    return False

def duplicate(img_path, factor, tag):
    relative = img_path.relative_to(img_dir)
    label_path = label_dir / relative.with_suffix(".txt")

    if not label_path.exists():
        return

    # Đọc nhãn gốc
    with open(label_path, "r") as f:
        labels = f.readlines()

    # Đọc ảnh gốc
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return

    # Danh sách các kiểu biến đổi để chọn ngẫu nhiên
    aug_types = ["flip", "color", "noise", "blur"]

    for i in range(factor):
        new_img_path = img_path.parent / f"{img_path.stem}_{tag}_{i}{img_path.suffix}"
        new_lbl_path = label_path.parent / f"{label_path.stem}_{tag}_{i}.txt"
        
        new_lbl_path.parent.mkdir(parents=True, exist_ok=True)

        # Chọn ngẫu nhiên 1 loại augmentation để áp dụng cho bản copy này
        aug_type = random.choice(aug_types)
        aug_img, aug_labels = apply_augmentation(img.copy(), labels, aug_type)

        # Lưu ảnh và nhãn đã biến đổi
        cv2.imwrite(str(new_img_path), aug_img)
        with open(new_lbl_path, "w") as f:
            f.write("\n".join([line.strip() for line in aug_labels if line.strip()]) + "\n")

# =========================
# oversample
# =========================
if __name__ == "__main__":
    print(f"[INFO] Bat dau Oversampling voi Data Augmentation...")
    glass_count = 0
    paper_count = 0
    metal_count = 0

    if not img_dir.exists():
        print(f"[ERROR] Khong tim thay thu muc: {img_dir}. Hay chay split_dataset.py truoc.")
        exit(1)

    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        relative = img_path.relative_to(img_dir)
        label_path = label_dir / relative.with_suffix(".txt")

        has_glass = contains_class(label_path, GLASS_ID)
        has_paper = contains_class(label_path, PAPER_ID)
        has_metal = contains_class(label_path, METAL_ID)

        if has_glass:
            duplicate(img_path, glass_factor, "glass")
            glass_count += 1

        if has_paper:
            duplicate(img_path, paper_factor, "paper")
            paper_count += 1
            
        if has_metal:
            duplicate(img_path, metal_factor, "metal")
            metal_count += 1

    print(f"Oversampled Glass images: {glass_count} (x{glass_factor} = {glass_count * glass_factor} copies)")
    print(f"Oversampled Paper images: {paper_count} (x{paper_factor} = {paper_count * paper_factor} copies)")
    print(f"Oversampled Metal images: {metal_count} (x{metal_factor} = {metal_count * metal_factor} copies)")
    print("[INFO] Done.")