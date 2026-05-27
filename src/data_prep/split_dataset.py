from pathlib import Path
import shutil
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

# ======================================
# config
# ======================================
base = Path("../../data/processed")

img_root = base / "images"
label_root = base / "labels"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

VALID_EXT = [".jpg", ".jpeg", ".png",".JPG"]


# ======================================
# helper
# ======================================
def get_all_images():
    images = []

    for p in img_root.rglob("*"):
        if not p.is_file():
            continue

        # ignore split folders
        if "train" in p.parts or "val" in p.parts or "test" in p.parts:
            continue

        # extension insensitive
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images.append(p)

    return images


def infer_num_classes(images):
    max_cls = -1

    for img_path in images:
        relative = img_path.relative_to(img_root)
        label_path = label_root / relative.with_suffix(".txt")

        if not label_path.exists():
            continue

        with open(label_path) as f:
            for line in f:
                cls = int(line.split()[0])
                max_cls = max(max_cls, cls)

    return max_cls + 1


def build_multilabel(images, num_classes):
    X = []
    Y = []

    for img_path in images:
        relative = img_path.relative_to(img_root)
        label_path = label_root / relative.with_suffix(".txt")

        if not label_path.exists():
            continue

        vec = np.zeros(num_classes)

        with open(label_path) as f:
            for line in f:
                cls = int(line.split()[0])
                vec[cls] = 1

        X.append(str(img_path))
        Y.append(vec)

    return np.array(X).reshape(-1, 1), np.array(Y)


def move_files(image_paths, split_name):
    for img_str in image_paths:
        img_path = Path(img_str)

        relative = img_path.relative_to(img_root)
        label_path = label_root / relative.with_suffix(".txt")

        # destination
        new_img = img_root / split_name / relative
        new_lbl = label_root / split_name / relative.with_suffix(".txt")

        new_img.parent.mkdir(parents=True, exist_ok=True)
        new_lbl.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(img_path), str(new_img))
        shutil.move(str(label_path), str(new_lbl))


def count_split(split):
    imgs = []
    for ext in VALID_EXT:
        imgs.extend((img_root / split).rglob(f"*{ext}"))

    lbls = list((label_root / split).rglob("*.txt"))

    return len(imgs), len(lbls)


# ======================================
# main
# ======================================
images = get_all_images()

print(f"Found {len(images)} images")

if len(images) == 0:
    raise ValueError("No images found")

num_classes = infer_num_classes(images)
print("num_classes =", num_classes)

X, Y = build_multilabel(images, num_classes)

# =========================
# step 1: train vs temp
# =========================
X_train, Y_train, X_temp, Y_temp = iterative_train_test_split(
    X,
    Y,
    test_size=(1 - train_ratio)
)

# =========================
# step 2: temp -> val/test
# =========================
X_val, Y_val, X_test, Y_test = iterative_train_test_split(
    X_temp,
    Y_temp,
    test_size=0.5
)

train_imgs = X_train.flatten()
val_imgs = X_val.flatten()
test_imgs = X_test.flatten()

print("\nSplit result")
print("Train:", len(train_imgs))
print("Val  :", len(val_imgs))
print("Test :", len(test_imgs))

# move
move_files(train_imgs, "train")
move_files(val_imgs, "val")
move_files(test_imgs, "test")

def remove_empty_dirs(root):
    # duyệt từ dưới lên để xóa folder con trước
    for p in sorted(root.rglob("*"), reverse=True):
        if p.is_dir():
            try:
                p.rmdir()   # chỉ xóa nếu rỗng
                print(f"Removed empty dir: {p}")
            except OSError:
                pass


remove_empty_dirs(img_root)
remove_empty_dirs(label_root)

print("\nDone.")

# verify
print("\nVerification")
for split in ["train", "val", "test"]:
    n_img, n_lbl = count_split(split)
    print(f"{split:5s}: images={n_img}, labels={n_lbl}")
