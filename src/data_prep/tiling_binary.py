import math
import random
import shutil
from pathlib import Path

import cv2

# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_BASE = BASE_DIR / "data" / "processed_binary"
OUTPUT_BASE = BASE_DIR / "data" / "processed_binary_tiled"

TILE_SIZE = 640
OVERLAP = 0.2

# --- Anti-overfitting controls ---
# INCLUDE_EMPTY      : True  = giữ TẤT CẢ tiles trống (nguy cơ bùng nổ dữ liệu!)
#                      False = bỏ hoàn toàn tiles trống (chỉ giữ tiles có object)
INCLUDE_EMPTY = False

# MAX_EMPTY_RATIO    : Chỉ có hiệu lực khi INCLUDE_EMPTY = True.
#                      Tỷ lệ tối đa tiles trống so với tiles có object.
#                      Ví dụ: 0.5 → tối đa 1 tile trống cho mỗi 2 tiles có object.
#                      None  → không giới hạn (giống hành vi cũ).
#                      Chỉ áp dụng cho split "train"; val/test giữ nguyên.
MAX_EMPTY_RATIO = 0.5

# EMPTY_SAMPLE_SEED  : Seed cho random sampling tiles trống (đảm bảo reproducibility)
EMPTY_SAMPLE_SEED = 42

# --- Box filter ---
MIN_BOX_SIZE_PX = 2
MIN_BOX_AREA_PX = 20
MIN_BOX_AREA_RATIO = 0.10

CLEAN_OUTPUT = False  # set True to reset output folder

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_yolo_labels(label_path, img_w, img_h):
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            x1 = (x - w / 2.0) * img_w
            y1 = (y - h / 2.0) * img_h
            x2 = (x + w / 2.0) * img_w
            y2 = (y + h / 2.0) * img_h

            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            labels.append((cls, x1, y1, x2, y2, area))

    return labels


def save_yolo_labels(label_path, labels, tile_w, tile_h):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, x1, y1, x2, y2 in labels:
            cx = (x1 + x2) / 2.0 / tile_w
            cy = (y1 + y2) / 2.0 / tile_h
            w = (x2 - x1) / tile_w
            h = (y2 - y1) / tile_h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def build_grid_positions(img_w, img_h, tile_size, overlap):
    if overlap < 0 or overlap >= 1:
        raise ValueError("OVERLAP must be in [0, 1).")

    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        raise ValueError("Invalid stride. Reduce OVERLAP or increase TILE_SIZE.")

    if img_w <= tile_size:
        xs = [0]
    else:
        xs = list(range(0, img_w - tile_size, stride)) + [img_w - tile_size]

    if img_h <= tile_size:
        ys = [0]
    else:
        ys = list(range(0, img_h - tile_size, stride)) + [img_h - tile_size]

    return xs, ys


def clip_box_to_tile(box, x0, y0, x1, y1):
    cls, bx1, by1, bx2, by2, area = box
    ix1 = max(bx1, x0)
    iy1 = max(by1, y0)
    ix2 = min(bx2, x1)
    iy2 = min(by2, y1)

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return (cls, ix1, iy1, ix2, iy2, area)


def tile_one_image(img_path, label_path, out_img_dir, out_lbl_dir,
                   apply_empty_cap=False, rng=None):
    """
    Tiling một ảnh thành nhiều tiles 640×640.

    Parameters
    ----------
    apply_empty_cap : bool
        Nếu True và MAX_EMPTY_RATIO không None, áp dụng giới hạn
        số tiles trống theo tỷ lệ MAX_EMPTY_RATIO so với tiles có object.
    rng : random.Random | None
        Random generator dùng cho sampling tiles trống (reproducible).

    Returns
    -------
    (n_with_obj, n_empty_saved, n_empty_skipped)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return 0, 0, 0

    img_h, img_w = img.shape[:2]
    labels = load_yolo_labels(label_path, img_w, img_h)

    xs, ys = build_grid_positions(img_w, img_h, TILE_SIZE, OVERLAP)

    # --- Pass 1: phân loại tiles thành có-object và trống ---
    tiles_with_obj = []   # list of (yi, xi, x0, y0, x1, y1, clipped_labels)
    tiles_empty    = []   # list of (yi, xi, x0, y0, x1, y1)

    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            x1 = min(x0 + TILE_SIZE, img_w)
            y1 = min(y0 + TILE_SIZE, img_h)

            clipped_labels = []
            for box in labels:
                clipped = clip_box_to_tile(box, x0, y0, x1, y1)
                if clipped is None:
                    continue
                cls, bx1, by1, bx2, by2, orig_area = clipped
                clipped_w = bx2 - bx1
                clipped_h = by2 - by1
                clipped_area = clipped_w * clipped_h
                if clipped_w < MIN_BOX_SIZE_PX or clipped_h < MIN_BOX_SIZE_PX:
                    continue
                if clipped_area < MIN_BOX_AREA_PX:
                    continue
                if orig_area > 0 and (clipped_area / orig_area) < MIN_BOX_AREA_RATIO:
                    continue
                clipped_labels.append((cls, bx1 - x0, by1 - y0, bx2 - x0, by2 - y0))

            if clipped_labels:
                tiles_with_obj.append((yi, xi, x0, y0, x1, y1, clipped_labels))
            else:
                tiles_empty.append((yi, xi, x0, y0, x1, y1))

    # --- Pass 2: quyết định tiles trống nào được lưu ---
    if not INCLUDE_EMPTY:
        # Không lấy tiles trống → chỉ lưu tiles có object
        empty_to_save = []
    elif apply_empty_cap and MAX_EMPTY_RATIO is not None:
        # Giới hạn theo tỷ lệ: max n_empty = floor(n_obj * MAX_EMPTY_RATIO)
        n_obj = len(tiles_with_obj)
        max_empty = math.floor(n_obj * MAX_EMPTY_RATIO)
        if len(tiles_empty) <= max_empty:
            empty_to_save = tiles_empty
        else:
            # Random sample (với seed đã set ở ngoài để reproducible)
            empty_to_save = (rng or random).sample(tiles_empty, max_empty)
    else:
        # INCLUDE_EMPTY = True, không cap → giữ tất cả (hành vi cũ)
        empty_to_save = tiles_empty

    # --- Pass 3: lưu tiles ---
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    def _save(yi, xi, x0, y0, x1, y1, clipped_labels):
        tile = img[y0:y1, x0:x1]
        tile_h, tile_w = tile.shape[:2]
        out_name     = f"{img_path.stem}_tile_{yi}_{xi}{img_path.suffix}"
        out_img_path = out_img_dir / out_name
        out_lbl_path = out_lbl_dir / f"{img_path.stem}_tile_{yi}_{xi}.txt"
        cv2.imwrite(str(out_img_path), tile)
        save_yolo_labels(out_lbl_path, clipped_labels, tile_w, tile_h)

    for entry in tiles_with_obj:
        yi, xi, x0, y0, x1, y1, clipped_labels = entry
        _save(yi, xi, x0, y0, x1, y1, clipped_labels)

    for entry in empty_to_save:
        yi, xi, x0, y0, x1, y1 = entry
        _save(yi, xi, x0, y0, x1, y1, [])

    n_empty_skipped = len(tiles_empty) - len(empty_to_save)
    return len(tiles_with_obj), len(empty_to_save), n_empty_skipped


def extract_names_block(yaml_path):
    if not yaml_path.exists():
        return None

    lines = yaml_path.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == "names:" or line.startswith("names:"):
            start = i
            break

    if start is None:
        return None

    return "\n".join(lines[start:]) + "\n"


def write_dataset_yaml(output_base, names_block):
    yaml_text = (
        f"path: {output_base}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
    )
    if names_block:
        yaml_text += names_block
    else:
        yaml_text += "names:\n"

    (output_base / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


# =========================
# Main
# =========================
if CLEAN_OUTPUT and OUTPUT_BASE.exists():
    shutil.rmtree(OUTPUT_BASE)

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

src_yaml = INPUT_BASE / "dataset.yaml"
names_block = extract_names_block(src_yaml)
write_dataset_yaml(OUTPUT_BASE, names_block)

# Random generator cho sampling tiles trống (chỉ dùng khi INCLUDE_EMPTY=True + MAX_EMPTY_RATIO)
rng = random.Random(EMPTY_SAMPLE_SEED)

print("=" * 65)
print("  TILING CONFIG")
print("=" * 65)
print(f"  TILE_SIZE        : {TILE_SIZE} px")
print(f"  OVERLAP          : {OVERLAP*100:.0f}%  (stride = {int(TILE_SIZE*(1-OVERLAP))} px)")
print(f"  INCLUDE_EMPTY    : {INCLUDE_EMPTY}")
if INCLUDE_EMPTY and MAX_EMPTY_RATIO is not None:
    print(f"  MAX_EMPTY_RATIO  : {MAX_EMPTY_RATIO}  (train only, seed={EMPTY_SAMPLE_SEED})")
elif INCLUDE_EMPTY:
    print(f"  MAX_EMPTY_RATIO  : None  (không giới hạn - nguy cơ bùng nổ dữ liệu!)")
print(f"  MIN_BOX_AREA_RATIO: {MIN_BOX_AREA_RATIO}")
print("=" * 65)

grand_total_orig  = 0
grand_total_tiles = 0

for split in SPLITS:
    src_img_root = INPUT_BASE / "images" / split
    src_lbl_root = INPUT_BASE / "labels" / split

    if not src_img_root.exists():
        print(f"[WARN] Missing split images: {src_img_root}")
        continue

    # Cap chỉ áp dụng cho train để tránh overfitting;
    # val/test giữ nguyên (cần đánh giá chính xác, không lọc bỏ)
    apply_cap = (split == "train")

    print(f"\n[INFO] Tiling split: {split}  (apply_empty_cap={apply_cap})")
    img_paths = [p for p in src_img_root.rglob("*") if p.suffix.lower() in IMG_EXTS]

    total_with_obj  = 0
    total_empty_saved   = 0
    total_empty_skipped = 0

    for img_path in img_paths:
        rel = img_path.relative_to(src_img_root)
        lbl_path = (src_lbl_root / rel).with_suffix(".txt")

        out_img_dir = OUTPUT_BASE / "images" / split / rel.parent
        out_lbl_dir = OUTPUT_BASE / "labels" / split / rel.parent

        n_obj, n_emp_saved, n_emp_skip = tile_one_image(
            img_path, lbl_path, out_img_dir, out_lbl_dir,
            apply_empty_cap=apply_cap, rng=rng,
        )
        total_with_obj      += n_obj
        total_empty_saved   += n_emp_saved
        total_empty_skipped += n_emp_skip

    total_tiles = total_with_obj + total_empty_saved
    expansion   = total_tiles / max(len(img_paths), 1)
    pct_empty   = total_empty_saved / max(total_tiles, 1) * 100

    print(f"  Ảnh gốc      : {len(img_paths):>6,}")
    print(f"  Tiles lưu    : {total_tiles:>6,}  ({expansion:.1f}x mở rộng)")
    print(f"    ├─ có object: {total_with_obj:>6,}")
    print(f"    ├─ trống (lưu)   : {total_empty_saved:>6,}  ({pct_empty:.1f}% của tổng tiles)")
    print(f"    └─ trống (bỏ qua): {total_empty_skipped:>6,}")

    grand_total_orig  += len(img_paths)
    grand_total_tiles += total_tiles

print(f"\n{'='*65}")
print(f"  TỔNG: {grand_total_orig:,} ảnh gốc → {grand_total_tiles:,} tiles")
print(f"{'='*65}")
print("Done.")
