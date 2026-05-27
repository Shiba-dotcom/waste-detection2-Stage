import json
import pandas as pd
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Uu tien dung file da lam sach, fallback sang file goc
cleaned_path = BASE_DIR / "data" / "raw" / "annotations_cleaned.json"
original_path = BASE_DIR / "data" / "raw" / "annotations.json"

if os.path.exists(cleaned_path):
    ann_path = cleaned_path
    print("[INFO] Su dung annotations_cleaned.json (da lam sach)")
else:
    ann_path = original_path
    print("[WARN] Khong tim thay annotations_cleaned.json, dung file goc")
    print("       Hay chay 'python src/data_cleaning.py' truoc!")

with open(ann_path, "r") as f:
    data = json.load(f)

mapping_df = pd.read_csv(BASE_DIR / "src" / "meta" / "mapping.csv")
mapping = dict(zip(mapping_df["name"], mapping_df["group"]))

idtn = {c["id"]: c["name"] for c in data["categories"]}
img_map = {img["id"]: img for img in data["images"]}
raw_img_dir = str(BASE_DIR / "data" / "raw")

# ═══════════════════════════════════════════════════════════════════
# CHẠY CẢ 2 CHẾ ĐỘ CÙNG LÚC (2-Stage Pipeline)
# ═══════════════════════════════════════════════════════════════════
# Vòng lặp chạy 2 lần:
# Lần 1 (False): Tạo data/processed/ (5 classes) -> Phục vụ Stage 2 cắt ảnh.
# Lần 2 (True):  Tạo data/processed_binary/ (1 class Waste) -> Phục vụ Stage 1 YOLO.

for BINARY_MODE in [False, True]:
    mode_str = "BINARY (1 class: Waste)" if BINARY_MODE else "MULTI-CLASS (5 classes)"
    print(f"\n[INFO] Đang xử lý chế độ: {mode_str}...")

    # Chon thu muc output tuy theo che do
    if BINARY_MODE:
        base_out = str(BASE_DIR / "data" / "processed_binary")
        classes = ["Waste"]
        class2id = {"Waste": 0}
    else:
        base_out = str(BASE_DIR / "data" / "processed")
        classes = sorted(set(mapping.values()))
        class2id = {c: i for i, c in enumerate(classes)}

    img_out = os.path.join(base_out, "images") 
    label_out = os.path.join(base_out, "labels") 

    if os.path.exists(base_out):
        shutil.rmtree(base_out)

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)

    # 1. Copy hình ảnh
    print("       Đang copy hình ảnh...")
    for img_id, img in img_map.items():
        src = os.path.join(raw_img_dir, img["file_name"])
        dst = os.path.join(img_out, img["file_name"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)

    # 2. Tạo nhãn (Labels)
    print("       Đang tạo file nhãn YOLO...")
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_info = img_map[img_id]
        img_name = os.path.splitext(img_info["file_name"])[0]

        cat_name = idtn[ann["category_id"]]
        label = mapping.get(cat_name, "Other")

        if BINARY_MODE:
            class_id = 0
        else:
            class_id = class2id[label]

        x, y, w, h = ann["bbox"]
        img_w, img_h = img_info["width"], img_info["height"]

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        txt_path = os.path.join(label_out, f"{img_name}.txt")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)

        with open(txt_path, "a") as f:
            f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

    # 3. Tạo file dataset.yaml
    yaml_path = os.path.join(base_out, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {base_out}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"       => Đã xử lý xong ({mode_str}).")

print("\n[HOÀN TẤT] Cả 2 bộ dữ liệu đã được tạo thành công!")



"""

[5] 
shutil.rmtree(base_out) có tác dụng reset tránh giữ lại dữ liệu cũ gấy sai lệch kết quả

[7]
Copy sang để tránh thay đổi dataset, cũng như là tách biệt dữ liệu gốc với dữ liêuj tiền xử lý.


[9]
cat_name = idtn[ann["category_id"]]
label = mapping.get(cat_name, "Other")
class_id = class2id[label]
    
Trong đó:
ann["category_id"] lấy mã lớp của object hiện tại từ annotation.

idtn: id to name
idtn = {c["id"]: c["name"] for c in data["categories"]} : lấy tên lớp thật tương ứng với category_id của object hiện tại.

label = mapping.get(cat_name, "Other")
nếu tìm thấy class trong mapping thì lấy lớp tương ứng
Nếu không tìm thấy thì mặc định gán vào lớp "Other"

class_id = class2id[label] chuyển nhãn dán (label) về id tương ứng do YOLO không đọc chữ, mà là nhận số nguyên
"""

# TODO: thống kê số mẫu mỗi lớp
# train/val split
# baseline model