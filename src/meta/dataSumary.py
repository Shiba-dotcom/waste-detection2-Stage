from pathlib import Path
from collections import Counter

# =========================
# config
# =========================
base = Path("../../data/processed")

splits = ["train", "val","test"]

CLASS_NAMES = {
    0: "Glass",
    1: "Metal",
    2: "Other",
    3: "Paper",
    4: "Plastic"
}


def summarize_split(split):
    img_dir = base / "images" / split
    label_dir = base / "labels" / split

    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(img_dir.rglob(ext))

    labels = list(label_dir.rglob("*.txt"))

    object_counter = Counter()
    image_counter = Counter()

    for label_path in labels:
        classes_in_image = set()

        with open(label_path) as f:
            for line in f:
                cls = int(line.split()[0])

                object_counter[cls] += 1
                classes_in_image.add(cls)

        for cls in classes_in_image:
            image_counter[cls] += 1

    return imgs, labels, object_counter, image_counter


# =========================
# summary
# =========================
for split in splits:
    imgs, labels, obj_cnt, img_cnt = summarize_split(split)

    print("\n" + "="*60)
    print(split.upper())
    print("="*60)

    print(f"Images : {len(imgs)}")
    print(f"Labels : {len(labels)}")

    max_obj = max(obj_cnt.values()) if obj_cnt else 1

    print("\nClass distribution:")
    print("-"*60)

    for cls_id, name in CLASS_NAMES.items():
        obj = obj_cnt[cls_id]
        img = img_cnt[cls_id]
        ratio = obj / max_obj

        print(
            f"{name:10s} | "
            f"Objects: {obj:5d} | "
            f"Images: {img:5d} | "
            f"Ratio: {ratio:.2f}"
        )

print("\nDone.")