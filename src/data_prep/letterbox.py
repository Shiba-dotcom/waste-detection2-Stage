import os
import glob
import cv2
from pathlib import Path

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TARGET_SIZE = 640
PAD_COLOR = (114, 114, 114)  # Màu xám tiêu chuẩn của YOLO

def letterbox_image_and_labels(img_path, label_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Khong the doc anh: {img_path}")
        return False
        
    h, w = img.shape[:2]
    
    # Tính scale ratio (r) để giữ nguyên Aspect Ratio
    r = min(TARGET_SIZE / w, TARGET_SIZE / h)
    
    # Tính kích thước ảnh sau khi scale (chưa pad)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    
    # Tính lượng padding cần thiết để đạt TARGET_SIZE
    dw = (TARGET_SIZE - new_unpad_w) / 2  # padding hai bên
    dh = (TARGET_SIZE - new_unpad_h) / 2  # padding trên dưới
    
    # Chia đều padding (làm tròn)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Resize ảnh
    img_resized = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    
    # Thêm viền (Padding)
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=PAD_COLOR)
    
    # Ghi đè lại ảnh đã được letterbox
    cv2.imwrite(str(img_path), img_padded)
    
    # Cập nhật tọa độ bounding box nếu có file nhãn
    if label_path.exists():
        new_labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
                
                # Chuyển từ normalize (0-1) sang pixel gốc
                abs_cx = cx * w
                abs_cy = cy * h
                abs_bw = bw * w
                abs_bh = bh * h
                
                # Áp dụng scale và cộng thêm phần padding
                new_cx = abs_cx * r + left
                new_cy = abs_cy * r + top
                new_bw = abs_bw * r
                new_bh = abs_bh * r
                
                # Chuyển lại về normalize (0-1) cho ảnh mới (TARGET_SIZE x TARGET_SIZE)
                norm_cx = new_cx / TARGET_SIZE
                norm_cy = new_cy / TARGET_SIZE
                norm_bw = new_bw / TARGET_SIZE
                norm_bh = new_bh / TARGET_SIZE
                
                new_labels.append(f"{cls} {norm_cx:.6f} {norm_cy:.6f} {norm_bw:.6f} {norm_bh:.6f}")
                
        # Ghi đè lại file nhãn
        with open(label_path, "w") as f:
            f.write("\n".join(new_labels) + "\n")
            
    return True

if __name__ == "__main__":
    print(f"[INFO] Bat dau ap dung Letterbox Padding ({TARGET_SIZE}x{TARGET_SIZE}) cho toan bo dataset...")
    
    img_dir = PROCESSED_DIR / "images"
    lbl_dir = PROCESSED_DIR / "labels"
    
    if not img_dir.exists():
        print(f"[ERROR] Khong tim thay thu muc: {img_dir}. Hay chay split_dataset.py truoc.")
        exit(1)
        
    # Lấy danh sách toàn bộ ảnh trong các tập train, val, test
    all_imgs = glob.glob(str(img_dir / "**" / "*.jpg"), recursive=True)
    all_imgs += glob.glob(str(img_dir / "**" / "*.png"), recursive=True)
    
    count = 0
    for img_path_str in all_imgs:
        img_path = Path(img_path_str)
        # Đường dẫn nhãn tương ứng
        rel_path = img_path.relative_to(img_dir)
        lbl_path = (lbl_dir / rel_path).with_suffix(".txt")
        
        if letterbox_image_and_labels(img_path, lbl_path):
            count += 1
            
    print(f"[INFO] Hoan thanh! Da xu ly letterbox cho {count} anh va cap nhat file nhãn (.txt) tuong ung.")