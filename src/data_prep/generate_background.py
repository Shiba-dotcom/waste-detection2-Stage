"""
generate_background.py - Tạo class Background (ảnh nhiễu) cho Classifier Stage 2
Nhóm 2 - Waste Detection (2-Stage Pipeline)

Script này đọc ảnh từ dataset gốc (processed_binary), sinh ra các random crops
không đè lên bất kỳ rác nào (Ground Truth) để làm ảnh Background (nền).
Mục tiêu: Dạy cho EfficientNet-B2 ở Stage 2 biết cách "từ chối" các False Positives
do YOLO sinh ra ở Stage 1 (ví dụ crop nhầm vào lá cây, mặt đường, tay người...).

Chạy SAU: merge_external_datasets.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import cv2
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

# =========================
# Config
# =========================
random.seed(42)
np.random.seed(42)

# Tự động phát hiện môi trường
import os as _os
ON_KAGGLE = _os.environ.get("ON_KAGGLE", "0") == "1" or _os.path.exists("/kaggle/working")

if ON_KAGGLE:
    BINARY_DIR = Path("/kaggle/working/waste-detection2-Stage/data/processed_binary")
    OUTPUT_DIR = Path("/kaggle/working/waste-detection2-Stage/data/classification_merged")
else:
    BASE_DIR = Path(__file__).resolve().parents[2]
    BINARY_DIR = BASE_DIR / "data" / "processed_binary"
    OUTPUT_DIR = BASE_DIR / "data" / "classification_merged"

# Số lượng ảnh Background mục tiêu cho mỗi split
TARGET_COUNTS = {
    "train": 2500,  # Bằng MAX_PER_CLASS ở merge_external_datasets.py
    "val": 400,
    "test": 400
}

# Kích thước crop (tương đối so với ảnh gốc)
MIN_CROP_RATIO = 0.1
MAX_CROP_RATIO = 0.4
MAX_ATTEMPTS = 20  # Số lần thử tìm crop không trùng rác trên 1 ảnh

# =========================
# Helper Functions
# =========================
def parse_yolo_labels(lbl_path, img_w, img_h):
    """Đọc file YOLO label và trả về danh sách [x1, y1, x2, y2]."""
    gt_boxes = []
    if not lbl_path.exists():
        return gt_boxes
        
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            gt_boxes.append([x1, y1, x2, y2])
            
    return gt_boxes

def compute_iou(box1, box2):
    """Tính Intersection over Union (IoU) giữa 2 bounding box."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_w = max(0, x2_inter - x1_inter)
    inter_h = max(0, y2_inter - y1_inter)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

# =========================
# Main Logic
# =========================
if __name__ == "__main__":
    print("=" * 65)
    print("  GENERATE BACKGROUND CLASS - Stage 2")
    print("=" * 65)
    print(f"  Source (Binary) : {BINARY_DIR}")
    print(f"  Output (Merged) : {OUTPUT_DIR}")
    print("=" * 65)

    if not BINARY_DIR.exists():
        print(f"[LỖI] Không tìm thấy dữ liệu nguồn: {BINARY_DIR}")
        sys.exit(1)

    for split in ["train", "val", "test"]:
        print(f"\n[XỬ LÝ] Split: {split.upper()}")
        
        img_dir = BINARY_DIR / "images" / split
        lbl_dir = BINARY_DIR / "labels" / split
        out_bg_dir = OUTPUT_DIR / split / "Background"
        out_bg_dir.mkdir(parents=True, exist_ok=True)
        
        if not img_dir.exists():
            print(f"  [WARN] Không có thư mục {img_dir}, bỏ qua.")
            continue
            
        img_paths = list(img_dir.rglob("*.[jp][pn]g"))
        random.shuffle(img_paths)
        
        target = TARGET_COUNTS.get(split, 500)
        generated_count = 0
        
        pbar = tqdm(total=target, desc=f"Generating {split}", ncols=100)
        
        # Lặp vô hạn qua các ảnh cho đến khi đủ số lượng target
        idx = 0
        while generated_count < target and len(img_paths) > 0:
            img_path = img_paths[idx % len(img_paths)]
            idx += 1
            
            # Tính đường dẫn label
            rel_path = img_path.relative_to(img_dir)
            lbl_path = lbl_dir / rel_path.with_suffix(".txt")
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            gt_boxes = parse_yolo_labels(lbl_path, w, h)
            
            # Cố gắng sinh 1 ảnh nền hợp lệ từ ảnh này
            for _ in range(MAX_ATTEMPTS):
                # 1. Random kích thước crop
                crop_w = int(w * random.uniform(MIN_CROP_RATIO, MAX_CROP_RATIO))
                crop_h = int(h * random.uniform(MIN_CROP_RATIO, MAX_CROP_RATIO))
                
                if crop_w < 30 or crop_h < 30: continue
                
                # 2. Random tọa độ
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
                x2 = x1 + crop_w
                y2 = y1 + crop_h
                crop_box = [x1, y1, x2, y2]
                
                # 3. Kiểm tra IoU với các rác (GT)
                # Đảm bảo không dính rác, hoặc dính rất ít (IoU < 0.05)
                is_valid = True
                for gt in gt_boxes:
                    if compute_iou(crop_box, gt) > 0.05:
                        is_valid = False
                        break
                        
                if is_valid:
                    # 4. Lưu ảnh
                    crop_img = img[y1:y2, x1:x2]
                    out_name = f"bg_{img_path.stem}_{x1}_{y1}.jpg"
                    cv2.imwrite(str(out_bg_dir / out_name), crop_img)
                    
                    generated_count += 1
                    pbar.update(1)
                    break # Chỉ lấy 1 crop thành công mỗi lần xét ảnh để đa dạng nền
                    
        pbar.close()
        print(f"  -> Đã tạo {generated_count}/{target} ảnh Background cho {split}.")

    print(f"\n{'=' * 65}")
    print("  [HOÀN TẤT] Sinh dữ liệu Background thành công!")
    print(f"  Hãy cập nhật train_stage2_classifier.py đổi NUM_CLASSES = 6")
    print(f"  Và CLASS_NAMES = ['Background', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic']")
    print(f"{'=' * 65}")
