# %% [markdown]
# # Stage 1 – YOLOv8s Binary Waste Detector
# Huấn luyện YOLOv8s phát hiện rác (1 lớp duy nhất: "Waste").
# Script tương thích Kaggle Notebook (dùng `# %%` cell markers).

# %%
# ============================================================
# Cell 1: Cài đặt thư viện & Import
# ============================================================
# Trên Kaggle, cần cài ultralytics trước khi import.
# Ở local, đảm bảo đã cài sẵn trong virtualenv.

import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

try:
    import ultralytics
except ImportError:
    install("ultralytics")

import io, os, shutil
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Hỗ trợ in tiếng Việt trên mọi console (tránh UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print(f"[INFO] ultralytics version: {ultralytics.__version__}")

# %%
# ============================================================
# Cell 2: Cấu hình đường dẫn & Siêu tham số (Hyperparameters)
# ============================================================
# Chuyển ON_KAGGLE = True khi chạy trên Kaggle Notebook.
# Ở local, giữ ON_KAGGLE = False.

ON_KAGGLE = False

if ON_KAGGLE:
    # Kaggle: dữ liệu input ở /kaggle/input/, output ở /kaggle/working/
    DATA_DIR    = Path("/kaggle/input/waste-binary-tiled")
    DATASET_YAML = DATA_DIR / "dataset.yaml"
    PROJECT_DIR = Path("/kaggle/working/yolo_runs")
else:
    # Local: dùng thư mục processed_binary_tiled đã có sẵn
    BASE_DIR     = Path(__file__).resolve().parents[1]
    DATA_DIR     = BASE_DIR / "data" / "processed_binary_tiled"
    DATASET_YAML = DATA_DIR / "dataset.yaml"
    PROJECT_DIR  = BASE_DIR / "results" / "yolo_runs"

# ----- Siêu tham số huấn luyện -----
# YOLOv8s: cân bằng giữa tốc độ và độ chính xác, phù hợp cho detection.
# imgsz=640: kích thước input chuẩn, khớp với tile_size=640 trong pipeline tiling.
# cos_lr=True: Cosine Annealing LR giúp hội tụ mượt hơn so với StepLR.
# patience=20: early stopping nếu mAP không cải thiện sau 20 epoch.
MODEL_WEIGHTS = "yolov8s.pt"
IMG_SIZE      = 640
EPOCHS        = 100
BATCH_SIZE    = 16
PATIENCE      = 20
OPTIMIZER     = "auto"      # YOLO tự chọn optimizer phù hợp
LR0           = 0.01        # Learning rate ban đầu
COS_LR        = True        # Cosine Annealing scheduler
AUGMENT       = True        # Bật augmentation mặc định của YOLO
WORKERS       = 4           # Số worker cho DataLoader
RUN_NAME      = "stage1_binary_yolov8s"

print(f"[INFO] Dataset YAML : {DATASET_YAML}")
print(f"[INFO] Project dir  : {PROJECT_DIR}")
print(f"[INFO] ON_KAGGLE    : {ON_KAGGLE}")

# %%
# ============================================================
# Cell 3: Kiểm tra dataset trước khi train
# ============================================================
# Đảm bảo file dataset.yaml tồn tại và in nội dung để xác nhận
# cấu hình đường dẫn + số lớp (chỉ 1 lớp: Waste).

assert DATASET_YAML.exists(), (
    f"[LỖI] Không tìm thấy dataset.yaml tại: {DATASET_YAML}\n"
    f"Hãy chạy pipeline tiền xử lý trước (tiling.py → split)."
)

print("=" * 60)
print("  NỘI DUNG dataset.yaml")
print("=" * 60)
print(DATASET_YAML.read_text(encoding="utf-8"))
print("=" * 60)

# Đếm số ảnh trong mỗi split để đảm bảo dữ liệu đầy đủ
for split in ["train", "val", "test"]:
    img_dir = DATA_DIR / "images" / split
    if img_dir.exists():
        n_imgs = len(list(img_dir.rglob("*.[jp][pn]g")))
        print(f"  {split:>5s}: {n_imgs:>6,} ảnh")
    else:
        print(f"  {split:>5s}: [KHÔNG TÌM THẤY THƯ MỤC]")

# %%
# ============================================================
# Cell 4: Huấn luyện YOLO
# ============================================================
# Khởi tạo model từ pretrained weights (COCO) rồi fine-tune
# trên dataset binary (1 lớp Waste).
# YOLO tự động lưu best.pt và last.pt vào thư mục runs.

model = YOLO(MODEL_WEIGHTS)

results = model.train(
    data       = str(DATASET_YAML),
    imgsz      = IMG_SIZE,
    epochs     = EPOCHS,
    batch      = BATCH_SIZE,
    patience   = PATIENCE,
    optimizer  = OPTIMIZER,
    lr0        = LR0,
    cos_lr     = COS_LR,
    augment    = AUGMENT,
    workers    = WORKERS,
    project    = str(PROJECT_DIR),
    name       = RUN_NAME,
    exist_ok   = True,           # Ghi đè nếu thư mục đã tồn tại
    save       = True,           # Lưu checkpoint
    save_period = -1,            # Chỉ lưu best + last (tiết kiệm dung lượng)
    plots      = True,           # Tạo biểu đồ tự động (confusion_matrix, ...)
    verbose    = True,
)

print("\n[INFO] Huấn luyện hoàn tất!")

# %%
# ============================================================
# Cell 5: Đánh giá trên tập Validation
# ============================================================
# Chạy validation chính thức trên tập val để lấy các chỉ số:
# mAP50, mAP50-95, Precision, Recall.

# Load best weights vừa train xong
best_weights = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
assert best_weights.exists(), f"[LỖI] Không tìm thấy best.pt tại: {best_weights}"

best_model = YOLO(str(best_weights))

val_results = best_model.val(
    data    = str(DATASET_YAML),
    imgsz   = IMG_SIZE,
    batch   = BATCH_SIZE,
    workers = WORKERS,
    split   = "val",
    verbose = True,
)

# In kết quả chính
print("\n" + "=" * 60)
print("  KẾT QUẢ VALIDATION (best.pt)")
print("=" * 60)
print(f"  mAP50      : {val_results.box.map50:.4f}")
print(f"  mAP50-95   : {val_results.box.map:.4f}")
print(f"  Precision  : {val_results.box.mp:.4f}")
print(f"  Recall     : {val_results.box.mr:.4f}")
print("=" * 60)

# %%
# ============================================================
# Cell 6: Đánh giá trên tập Test (nếu có)
# ============================================================
# Kiểm tra hiệu suất trên tập test – tập dữ liệu chưa từng
# thấy trong quá trình huấn luyện, đánh giá khả năng tổng quát.

test_img_dir = DATA_DIR / "images" / "test"

if test_img_dir.exists() and any(test_img_dir.iterdir()):
    test_results = best_model.val(
        data    = str(DATASET_YAML),
        imgsz   = IMG_SIZE,
        batch   = BATCH_SIZE,
        workers = WORKERS,
        split   = "test",
        verbose = True,
    )

    print("\n" + "=" * 60)
    print("  KẾT QUẢ TEST (best.pt)")
    print("=" * 60)
    print(f"  mAP50      : {test_results.box.map50:.4f}")
    print(f"  mAP50-95   : {test_results.box.map:.4f}")
    print(f"  Precision  : {test_results.box.mp:.4f}")
    print(f"  Recall     : {test_results.box.mr:.4f}")
    print("=" * 60)
else:
    print("[WARN] Không tìm thấy tập test, bỏ qua đánh giá test.")

# %%
# ============================================================
# Cell 7: Vẽ biểu đồ Training Curves từ results.csv
# ============================================================
# YOLO tự động ghi log metrics mỗi epoch vào results.csv.
# Vẽ lại để phân tích quá trình hội tụ: loss giảm? mAP tăng?

results_csv = PROJECT_DIR / RUN_NAME / "results.csv"

if results_csv.exists():
    df = pd.read_csv(results_csv)
    # Chuẩn hóa tên cột (bỏ khoảng trắng thừa)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("YOLOv8s Binary Detector – Training Curves", fontsize=14, fontweight="bold")

    # --- Box Loss (train vs val) ---
    ax = axes[0, 0]
    if "train/box_loss" in df.columns:
        ax.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", color="tab:blue")
    if "val/box_loss" in df.columns:
        ax.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Box Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Cls Loss (train vs val) ---
    ax = axes[0, 1]
    if "train/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", color="tab:blue")
    if "val/cls_loss" in df.columns:
        ax.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Classification Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- mAP50 & mAP50-95 ---
    ax = axes[1, 0]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50", color="tab:green")
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95", color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Precision & Recall ---
    ax = axes[1, 1]
    if "metrics/precision(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/precision(B)"], label="Precision", color="tab:purple")
    if "metrics/recall(B)" in df.columns:
        ax.plot(df["epoch"], df["metrics/recall(B)"], label="Recall", color="tab:cyan")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Lưu biểu đồ
    chart_path = PROJECT_DIR / RUN_NAME / "training_curves.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    print(f"[INFO] Đã lưu biểu đồ: {chart_path}")
    plt.show()
else:
    print("[WARN] Không tìm thấy results.csv, không thể vẽ biểu đồ.")

# %%
# ============================================================
# Cell 8: Export model (ONNX) để triển khai
# ============================================================
# Export sang ONNX để sử dụng trong pipeline inference.
# ONNX cho phép chạy trên nhiều nền tảng (CPU, TensorRT, OpenVINO).

export_model = YOLO(str(best_weights))

export_path = export_model.export(
    format = "onnx",
    imgsz  = IMG_SIZE,
    half   = False,          # True nếu muốn FP16 (cần GPU hỗ trợ)
    simplify = True,         # Đơn giản hóa graph ONNX
)

print(f"\n[INFO] Model đã export sang ONNX: {export_path}")

# %%
# ============================================================
# Cell 9: Sao chép best weights về thư mục cố định
# ============================================================
# Copy best.pt ra ngoài thư mục runs để dễ truy cập.
# Trên Kaggle, copy vào /kaggle/working/ để download.

if ON_KAGGLE:
    final_dst = Path("/kaggle/working/stage1_best.pt")
else:
    final_dst = Path(__file__).resolve().parents[1] / "models" / "stage1_best.pt"

final_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(best_weights), str(final_dst))
print(f"[INFO] Đã copy best weights → {final_dst}")

# %%
# ============================================================
# Cell 10: Tổng kết
# ============================================================
print("\n" + "=" * 60)
print("  HOÀN TẤT HUẤN LUYỆN STAGE 1 – BINARY DETECTOR")
print("=" * 60)
print(f"  Model        : YOLOv8s (binary – 1 lớp: Waste)")
print(f"  Best weights : {best_weights}")
print(f"  Export ONNX  : {export_path}")
print(f"  Final copy   : {final_dst}")
print("=" * 60)
print("\n[NEXT] Tiếp theo: chạy train_stage2_classifier.py")
print("       để huấn luyện EfficientNet-B2 phân loại 5 lớp rác.")
