import json
import pandas as pd
import os
import shutil
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# CHE DO BINARY (2-Stage Pipeline)
# ═══════════════════════════════════════════════════════════════════
# True  = Gop tat ca class thanh 1 class "Waste" (cho Stage 1 Detection)
#          Output: data/processed_binary/
# False = Giu nguyen 5 classes (Glass, Metal, Other, Paper, Plastic)
#          Output: data/processed/
BINARY_MODE = False

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

mode_str = "BINARY (1 class: Waste)" if BINARY_MODE else "MULTI-CLASS (5 classes)"
print(f"[INFO] Che do: {mode_str}")

with open(ann_path, "r") as f:
    data = json.load(f)

mapping_df = pd.read_csv(BASE_DIR / "src" / "meta" / "mapping.csv")
mapping = dict(zip(mapping_df["name"], mapping_df["group"]))


idtn = {c["id"]: c["name"] for c in data["categories"]}

# Chon thu muc output tuy theo che do
if BINARY_MODE:
    base_out = str(BASE_DIR / "data" / "processed_binary")
else:
    base_out = str(BASE_DIR / "data" / "processed")

img_out = os.path.join(base_out, "images") # Thư mục chứa file hình
label_out = os.path.join(base_out, "labels") # Thư mục chứa file annotation (định dạng txt)

if os.path.exists(base_out):
    shutil.rmtree(base_out)


os.makedirs(img_out, exist_ok=True)
os.makedirs(label_out, exist_ok=True)

# Thiet lap classes tuy theo che do
if BINARY_MODE:
    # Chi 1 class duy nhat
    classes = ["Waste"]
    class2id = {"Waste": 0}
else:
    classes = sorted(set(mapping.values()))
    class2id = {c: i for i, c in enumerate(classes)}



img_map = {img["id"]: img for img in data["images"]}

raw_img_dir = str(BASE_DIR / "data" / "raw")

for img_id, img in img_map.items():
    src = os.path.join(raw_img_dir, img["file_name"])
    dst = os.path.join(img_out, img["file_name"])
    
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    
    shutil.copy(src, dst)


for ann in data["annotations"]:
    img_id = ann["image_id"]
    img_info = img_map[img_id]

    img_name = os.path.splitext(img_info["file_name"])[0]

    # [9] Mapping lại nhãn
    cat_name = idtn[ann["category_id"]]
    label = mapping.get(cat_name, "Other")

    # Trong BINARY_MODE: tat ca deu la class 0 ("Waste")
    if BINARY_MODE:
        class_id = 0
    else:
        class_id = class2id[label]

    x, y, w, h = ann["bbox"]

    img_w, img_h = img_info["width"], img_info["height"]

    # [8] Chuyển annotation (định dạng COCO) sang format YOLO
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    txt_path = os.path.join(label_out, f"{img_name}.txt")

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)

    with open(txt_path, "a") as f:
        f.write(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}\n")


yaml_path = os.path.join(base_out, "dataset.yaml")

with open(yaml_path, "w") as f:
    f.write(f"""
path: {base_out}
train: images/train
val: images/val
test: images/test

names:
""")
    for i, c in enumerate(classes):
        f.write(f"  {i}: {c}\n")

print(f"Da xu ly xong ({mode_str}).")



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