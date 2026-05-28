# %% [markdown]
# # Thí nghiệm Stage 1 – So sánh các chiến lược Inference
# 
# **Mục tiêu:** Đánh giá 5 cấu hình inference/model khác nhau,
# bao gồm YOLOv8s baseline và YOLO26s (mới nhất 2026).
#
# | TN | Model | Conf | SAHI | Mục đích |
# |----|-------|------|------|----------|
# | Baseline | YOLOv8s | 0.25 | ❌ | Mốc so sánh |
# | TN1 | YOLOv8s | 0.15 | ❌ | Đo tác động hạ confidence |
# | TN2 | YOLOv8s | 0.25 | ✅ | Đo tác động SAHI cho vật nhỏ |
# | TN3 | YOLOv8s | 0.15 | ✅ | Kết hợp cả hai |
# | TN4 | YOLO26s | 0.25 | ❌ | So sánh kiến trúc mới YOLO26 |

# %%
# ============================================================
# Cell 1: Cài đặt thư viện
# ============================================================
import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import ultralytics
except ImportError:
    install("ultralytics")

try:
    from sahi import AutoDetectionModel
except ImportError:
    install("sahi")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# %%
# ============================================================
# Cell 2: Import & Cấu hình đường dẫn
# ============================================================
import os
import cv2
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ---------- Phát hiện môi trường ----------
ON_KAGGLE = os.path.exists("/kaggle/working")

if ON_KAGGLE:
    PROJECT_DIR = Path("/kaggle/working/waste-detection2-Stage")
else:
    PROJECT_DIR = Path(__file__).resolve().parents[1]

# Đường dẫn dữ liệu
DATA_DIR     = PROJECT_DIR / "data" / "processed_binary"
VAL_IMG_DIR  = DATA_DIR / "images" / "val"
VAL_LBL_DIR  = DATA_DIR / "labels" / "val"

# Đường dẫn model YOLOv8s (đã train)
WEIGHTS_PATH = PROJECT_DIR / "stage1_best.pt"
if not WEIGHTS_PATH.exists():
    # Thử đường dẫn thay thế
    alt = PROJECT_DIR / "yolo_runs" / "stage1_binary_yolov8s" / "weights" / "best.pt"
    if alt.exists():
        WEIGHTS_PATH = alt

# YOLO26s – sẽ được train trong thí nghiệm
YOLO26_RUN_DIR    = PROJECT_DIR / "results" / "yolo26_runs"
YOLO26_WEIGHTS    = YOLO26_RUN_DIR / "stage1_yolo26s" / "weights" / "best.pt"
DATASET_YAML      = DATA_DIR / "dataset.yaml"

# Đường dẫn output
OUTPUT_DIR = PROJECT_DIR / "results" / "stage1_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[INFO] Môi trường     : {'Kaggle' if ON_KAGGLE else 'Local'}")
print(f"[INFO] YOLOv8s weights: {WEIGHTS_PATH}")
print(f"[INFO] Dataset YAML   : {DATASET_YAML}")
print(f"[INFO] Val images     : {VAL_IMG_DIR}")
print(f"[INFO] Val labels     : {VAL_LBL_DIR}")
print(f"[INFO] Output         : {OUTPUT_DIR}")

# Kiểm tra dữ liệu
assert WEIGHTS_PATH.exists(), f"[LỖI] Không tìm thấy weights: {WEIGHTS_PATH}"
assert VAL_IMG_DIR.exists(),  f"[LỖI] Không tìm thấy val images: {VAL_IMG_DIR}"
assert VAL_LBL_DIR.exists(),  f"[LỖI] Không tìm thấy val labels: {VAL_LBL_DIR}"

# %%
# ============================================================
# Cell 3: Hàm hỗ trợ – Đọc Ground Truth & Tính IoU
# ============================================================

def get_val_images():
    """Thu thập tất cả ảnh trong thư mục val."""
    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for p in sorted(VAL_IMG_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            imgs.append(p)
    return imgs

def parse_yolo_label(label_path, img_w, img_h):
    """Đọc YOLO label → list of [x1, y1, x2, y2] (pixel coords)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # class_id bỏ qua (binary nên luôn = 0)
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes

def compute_iou(box1, box2):
    """Tính IoU giữa 2 box [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def match_predictions(gt_boxes, pred_boxes, pred_confs, iou_thresh=0.5):
    """
    Matching greedy giữa GT và predictions.
    
    Returns:
        tp: số True Positive
        fp: số False Positive  
        fn: số False Negative
        matched_gt_indices: set các GT đã match
    """
    tp, fp = 0, 0
    gt_matched = [False] * len(gt_boxes)
    
    # Sắp xếp predictions theo confidence giảm dần
    if len(pred_boxes) > 0:
        sorted_indices = np.argsort(pred_confs)[::-1]
    else:
        sorted_indices = []
    
    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        best_iou = 0
        best_gt = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = j
        
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1
    
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn

# %%
# ============================================================
# Cell 4: Hàm chạy từng thí nghiệm
# ============================================================

def run_experiment_standard(model, images, conf_thresh):
    """Chạy inference YOLO chuẩn (không SAHI)."""
    all_tp, all_fp, all_fn = 0, 0, 0
    total_gt, total_pred = 0, 0
    
    # Phân tích theo kích thước
    size_stats = {"small": {"tp": 0, "fn": 0}, 
                  "medium": {"tp": 0, "fn": 0}, 
                  "large": {"tp": 0, "fn": 0}}
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Ground Truth
        rel = img_path.relative_to(VAL_IMG_DIR)
        lbl_path = VAL_LBL_DIR / rel.with_suffix(".txt")
        gt_boxes = parse_yolo_label(lbl_path, w, h)
        
        # Predictions
        results = model.predict(
            source=img,
            conf=conf_thresh,
            iou=0.7,
            verbose=False,
        )
        
        pred_boxes = []
        pred_confs = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                pred_boxes.append(xyxy.tolist())
                pred_confs.append(float(box.conf[0]))
        
        # Matching
        tp, fp, fn = match_predictions(gt_boxes, pred_boxes, pred_confs)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # Phân tích theo kích thước GT box
        gt_matched = [False] * len(gt_boxes)
        if len(pred_boxes) > 0:
            sorted_indices = np.argsort(pred_confs)[::-1]
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                best_iou, best_gt = 0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if gt_matched[j]: continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
                if best_iou >= 0.5 and best_gt >= 0:
                    gt_matched[best_gt] = True
        
        for j, gt_box in enumerate(gt_boxes):
            # Tính diện tích tương đối
            bw = (gt_box[2] - gt_box[0]) / w
            bh = (gt_box[3] - gt_box[1]) / h
            area_rel = bw * bh
            
            if area_rel < 0.01:        # < 1% ảnh → nhỏ
                cat = "small"
            elif area_rel < 0.05:       # 1-5% ảnh → trung bình
                cat = "medium"
            else:                       # > 5% ảnh → lớn
                cat = "large"
            
            if gt_matched[j]:
                size_stats[cat]["tp"] += 1
            else:
                size_stats[cat]["fn"] += 1
    
    return {
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "total_gt": total_gt, "total_pred": total_pred,
        "size_stats": size_stats
    }


def run_experiment_sahi(weights_path, images, conf_thresh,
                        slice_size=640, overlap_ratio=0.2):
    """Chạy inference SAHI (sliced)."""
    all_tp, all_fp, all_fn = 0, 0, 0
    total_gt, total_pred = 0, 0
    
    size_stats = {"small": {"tp": 0, "fn": 0},
                  "medium": {"tp": 0, "fn": 0},
                  "large": {"tp": 0, "fn": 0}}
    
    # Khởi tạo model SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(weights_path),
        confidence_threshold=conf_thresh,
        device="cuda" if ON_KAGGLE else "cpu",
    )
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Ground Truth
        rel = img_path.relative_to(VAL_IMG_DIR)
        lbl_path = VAL_LBL_DIR / rel.with_suffix(".txt")
        gt_boxes = parse_yolo_label(lbl_path, w, h)
        
        # SAHI Sliced Prediction
        result = get_sliced_prediction(
            image=str(img_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            verbose=0,
        )
        
        pred_boxes = []
        pred_confs = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            pred_boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
            pred_confs.append(pred.score.value)
        
        # Matching
        tp, fp, fn = match_predictions(gt_boxes, pred_boxes, pred_confs)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        
        # Phân tích theo kích thước
        gt_matched = [False] * len(gt_boxes)
        if len(pred_boxes) > 0:
            sorted_indices = np.argsort(pred_confs)[::-1]
            for idx in sorted_indices:
                pred_box = pred_boxes[idx]
                best_iou, best_gt = 0, -1
                for j, gt_box in enumerate(gt_boxes):
                    if gt_matched[j]: continue
                    iou = compute_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j
                if best_iou >= 0.5 and best_gt >= 0:
                    gt_matched[best_gt] = True
        
        for j, gt_box in enumerate(gt_boxes):
            bw = (gt_box[2] - gt_box[0]) / w
            bh = (gt_box[3] - gt_box[1]) / h
            area_rel = bw * bh
            if area_rel < 0.01:
                cat = "small"
            elif area_rel < 0.05:
                cat = "medium"
            else:
                cat = "large"
            if gt_matched[j]:
                size_stats[cat]["tp"] += 1
            else:
                size_stats[cat]["fn"] += 1
    
    return {
        "tp": all_tp, "fp": all_fp, "fn": all_fn,
        "total_gt": total_gt, "total_pred": total_pred,
        "size_stats": size_stats
    }


def calc_metrics(result):
    """Tính Precision, Recall, F1 từ TP/FP/FN."""
    tp, fp, fn = result["tp"], result["fp"], result["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "Precision": round(precision, 4),
        "Recall":    round(recall, 4),
        "F1":        round(f1, 4),
        "TP": tp, "FP": fp, "FN": fn,
        "Total GT": result["total_gt"],
        "Total Pred": result["total_pred"],
    }

# %%
# ============================================================
# Cell 5a: Train YOLO26s (nếu chưa có weights)
# ============================================================
# YOLO26s được train trên cùng dataset binary với cùng siêu tham số
# như YOLOv8s để so sánh công bằng giữa 2 kiến trúc.

if not YOLO26_WEIGHTS.exists():
    print("\n" + "=" * 60)
    print("  🚀 TRAIN YOLO26s – Kiến trúc mới nhất 2026")
    print("=" * 60)
    print("  YOLO26s chưa có weights → Bắt đầu train...")
    print(f"  Dataset: {DATASET_YAML}")
    
    yolo26_model = YOLO("yolo26s.pt")  # Tải pretrained YOLO26s từ Ultralytics
    
    yolo26_model.train(
        data       = str(DATASET_YAML),
        imgsz      = 640,
        epochs     = 100,
        batch      = 16,
        patience   = 20,
        optimizer  = "auto",
        lr0        = 0.01,
        cos_lr     = True,
        augment    = True,
        workers    = 4,
        project    = str(YOLO26_RUN_DIR),
        name       = "stage1_yolo26s",
        exist_ok   = True,
        save       = True,
        save_period = -1,
        plots      = True,
        verbose    = True,
    )
    print("\n[INFO] ✅ Train YOLO26s hoàn tất!")
else:
    print(f"\n[INFO] YOLO26s weights đã tồn tại: {YOLO26_WEIGHTS}")
    print("       → Bỏ qua bước train, dùng weights có sẵn.")

assert YOLO26_WEIGHTS.exists(), (
    f"[LỖI] Không tìm thấy YOLO26s weights sau khi train: {YOLO26_WEIGHTS}"
)

# %%
# ============================================================
# Cell 5b: Chạy 5 Thí nghiệm
# ============================================================

val_images = get_val_images()
print(f"\n[INFO] Tìm thấy {len(val_images)} ảnh val")
print("=" * 60)

# Load models
model_v8s   = YOLO(str(WEIGHTS_PATH))
model_yolo26 = YOLO(str(YOLO26_WEIGHTS))

# ---------- Định nghĩa thí nghiệm ----------
# "model": chỉ định dùng model nào (v8s hoặc yolo26)
experiments = {
    "Baseline":          {"conf": 0.25, "sahi": False, "model": "v8s"},
    "TN1_LowConf":       {"conf": 0.15, "sahi": False, "model": "v8s"},
    "TN2_SAHI":          {"conf": 0.25, "sahi": True,  "model": "v8s"},
    "TN3_SAHI+LowConf":  {"conf": 0.15, "sahi": True,  "model": "v8s"},
    "TN4_YOLO26s":       {"conf": 0.25, "sahi": False, "model": "yolo26"},
}

all_results = {}

for exp_name, config in experiments.items():
    # Chọn model và weights path
    if config["model"] == "yolo26":
        current_model = model_yolo26
        current_weights = YOLO26_WEIGHTS
        model_label = "YOLO26s"
    else:
        current_model = model_v8s
        current_weights = WEIGHTS_PATH
        model_label = "YOLOv8s"
    
    print(f"\n{'='*60}")
    print(f"  🧪 {exp_name}")
    print(f"     model={model_label}, conf={config['conf']}, SAHI={'✅' if config['sahi'] else '❌'}")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    if config["sahi"]:
        raw = run_experiment_sahi(
            current_weights, val_images, config["conf"],
            slice_size=640, overlap_ratio=0.2,
        )
    else:
        raw = run_experiment_standard(current_model, val_images, config["conf"])
    
    elapsed = time.time() - t_start
    metrics = calc_metrics(raw)
    metrics["Time (s)"] = round(elapsed, 1)
    metrics["size_stats"] = raw["size_stats"]
    
    all_results[exp_name] = metrics
    
    print(f"  Precision : {metrics['Precision']}")
    print(f"  Recall    : {metrics['Recall']}")
    print(f"  F1        : {metrics['F1']}")
    print(f"  TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']}")
    print(f"  Thời gian : {metrics['Time (s)']}s")

# %%
# ============================================================
# Cell 6: Bảng So sánh Tổng hợp
# ============================================================

print("\n" + "=" * 70)
print("  📊 BẢNG SO SÁNH TỔNG HỢP – STAGE 1 INFERENCE EXPERIMENTS")
print("=" * 70)

# Tạo DataFrame
rows = []
for name, m in all_results.items():
    rows.append({
        "Thí nghiệm": name,
        "Model": "YOLO26s" if experiments[name]["model"] == "yolo26" else "YOLOv8s",
        "Conf": experiments[name]["conf"],
        "SAHI": "✅" if experiments[name]["sahi"] else "❌",
        "Precision": m["Precision"],
        "Recall": m["Recall"],
        "F1": m["F1"],
        "TP": m["TP"],
        "FP": m["FP"],
        "FN": m["FN"],
        "Tổng Pred": m["Total Pred"],
        "Tổng GT": m["Total GT"],
        "Time (s)": m["Time (s)"],
    })

df_compare = pd.DataFrame(rows)
print(df_compare.to_string(index=False))

# Lưu CSV
csv_path = OUTPUT_DIR / "comparison_table.csv"
df_compare.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"\n[INFO] Đã lưu bảng so sánh: {csv_path}")

# %%
# ============================================================
# Cell 7: Phân tích theo Kích thước Vật thể
# ============================================================

print("\n" + "=" * 70)
print("  📏 PHÂN TÍCH RECALL THEO KÍCH THƯỚC VẬT THỂ")
print("=" * 70)

size_rows = []
for name, m in all_results.items():
    ss = m["size_stats"]
    for cat in ["small", "medium", "large"]:
        tp_s = ss[cat]["tp"]
        fn_s = ss[cat]["fn"]
        total = tp_s + fn_s
        recall_s = tp_s / total if total > 0 else 0
        size_rows.append({
            "Thí nghiệm": name,
            "Kích thước": cat,
            "Detected": tp_s,
            "Missed": fn_s,
            "Total": total,
            "Recall": round(recall_s, 4),
        })

df_size = pd.DataFrame(size_rows)

# Pivot table cho dễ đọc
pivot = df_size.pivot_table(
    index="Thí nghiệm", 
    columns="Kích thước", 
    values="Recall",
    aggfunc="first"
)[["small", "medium", "large"]]
print(pivot.round(4))

csv_size_path = OUTPUT_DIR / "size_analysis.csv"
df_size.to_csv(csv_size_path, index=False, encoding="utf-8-sig")
print(f"\n[INFO] Đã lưu phân tích kích thước: {csv_size_path}")

# %%
# ============================================================
# Cell 8: Vẽ biểu đồ so sánh
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Stage 1 Inference Experiments – Comparison (incl. YOLO26)", 
             fontsize=16, fontweight="bold", y=1.02)

exp_names  = list(all_results.keys())
short_names = ["Baseline\nv8s, 0.25", "TN1\nv8s, 0.15", 
               "TN2\nv8s+SAHI", "TN3\nSAHI+0.15",
               "TN4\nYOLO26s"]
colors = ["#4A90D9", "#7BC47F", "#E8943A", "#D94A6B", "#9B59B6"]

# --- Chart 1: Precision, Recall, F1 ---
ax = axes[0]
x = np.arange(len(exp_names))
width = 0.22

prec_vals   = [all_results[n]["Precision"] for n in exp_names]
recall_vals = [all_results[n]["Recall"] for n in exp_names]
f1_vals     = [all_results[n]["F1"] for n in exp_names]

bars1 = ax.bar(x - width, prec_vals, width, label="Precision", color="#4A90D9", alpha=0.85)
bars2 = ax.bar(x,         recall_vals, width, label="Recall", color="#7BC47F", alpha=0.85)
bars3 = ax.bar(x + width, f1_vals, width, label="F1", color="#E8943A", alpha=0.85)

# Thêm giá trị trên cột
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("")
ax.set_ylabel("Score")
ax.set_title("Precision / Recall / F1", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Chart 2: Recall theo kích thước ---
ax = axes[1]
small_recalls  = [all_results[n]["size_stats"]["small"]["tp"] / 
                  max(1, all_results[n]["size_stats"]["small"]["tp"] + 
                      all_results[n]["size_stats"]["small"]["fn"])
                  for n in exp_names]
medium_recalls = [all_results[n]["size_stats"]["medium"]["tp"] /
                  max(1, all_results[n]["size_stats"]["medium"]["tp"] + 
                      all_results[n]["size_stats"]["medium"]["fn"])
                  for n in exp_names]
large_recalls  = [all_results[n]["size_stats"]["large"]["tp"] /
                  max(1, all_results[n]["size_stats"]["large"]["tp"] + 
                      all_results[n]["size_stats"]["large"]["fn"])
                  for n in exp_names]

bars1 = ax.bar(x - width, small_recalls, width, label="Small (<1%)", color="#FF6B6B", alpha=0.85)
bars2 = ax.bar(x,         medium_recalls, width, label="Medium (1-5%)", color="#4ECDC4", alpha=0.85)
bars3 = ax.bar(x + width, large_recalls, width, label="Large (>5%)", color="#45B7D1", alpha=0.85)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_title("Recall theo Kích thước Vật thể", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("Recall")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Chart 3: TP / FP / FN stacked ---
ax = axes[2]
tp_vals = [all_results[n]["TP"] for n in exp_names]
fp_vals = [all_results[n]["FP"] for n in exp_names]
fn_vals = [all_results[n]["FN"] for n in exp_names]

ax.bar(x, tp_vals, width=0.6, label="TP (Đúng)", color="#7BC47F", alpha=0.85)
ax.bar(x, fp_vals, width=0.6, bottom=tp_vals, label="FP (Nhầm)", color="#E8943A", alpha=0.85)
ax.bar(x, fn_vals, width=0.6, 
       bottom=[t+f for t,f in zip(tp_vals, fp_vals)],
       label="FN (Bỏ sót)", color="#D94A6B", alpha=0.85)

# Hiển thị số liệu
for i in range(len(exp_names)):
    # TP
    ax.text(i, tp_vals[i]/2, str(tp_vals[i]), ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    # FP
    ax.text(i, tp_vals[i] + fp_vals[i]/2, str(fp_vals[i]), ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    # FN
    ax.text(i, tp_vals[i] + fp_vals[i] + fn_vals[i]/2, str(fn_vals[i]), 
            ha="center", va="center", fontsize=10, fontweight="bold", color="white")

ax.set_title("Phân phối TP / FP / FN", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("Số lượng boxes")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
chart_path = OUTPUT_DIR / "experiments_comparison.png"
plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
print(f"[INFO] Đã lưu biểu đồ: {chart_path}")
plt.show()

# %%
# ============================================================
# Cell 9: Tính toán Delta (cải thiện so với Baseline)
# ============================================================

print("\n" + "=" * 70)
print("  📈 CẢI THIỆN SO VỚI BASELINE")
print("=" * 70)

base = all_results["Baseline"]

delta_rows = []
for name in exp_names[1:]:  # Bỏ qua Baseline
    m = all_results[name]
    delta_rows.append({
        "Thí nghiệm": name,
        "ΔPrecision": round(m["Precision"] - base["Precision"], 4),
        "ΔRecall":    round(m["Recall"] - base["Recall"], 4),
        "ΔF1":        round(m["F1"] - base["F1"], 4),
        "ΔTP":        m["TP"] - base["TP"],
        "ΔFP":        m["FP"] - base["FP"],
        "ΔFN":        m["FN"] - base["FN"],
    })

df_delta = pd.DataFrame(delta_rows)
print(df_delta.to_string(index=False))

csv_delta_path = OUTPUT_DIR / "delta_vs_baseline.csv"
df_delta.to_csv(csv_delta_path, index=False, encoding="utf-8-sig")

# %%
# ============================================================
# Cell 10: Lưu kết quả JSON đầy đủ & Tổng kết
# ============================================================

# Lưu raw results (bỏ size_stats object cho JSON serialization)
save_data = {}
for name, m in all_results.items():
    save_data[name] = {k: v for k, v in m.items() if k != "size_stats"}
    save_data[name]["size_stats"] = {}
    for cat in ["small", "medium", "large"]:
        ss = m["size_stats"][cat]
        total = ss["tp"] + ss["fn"]
        save_data[name]["size_stats"][cat] = {
            "tp": ss["tp"], "fn": ss["fn"], "total": total,
            "recall": round(ss["tp"] / total, 4) if total > 0 else 0
        }

json_path = OUTPUT_DIR / "experiment_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False)

print(f"\n[INFO] Đã lưu kết quả JSON: {json_path}")

# ---------- Tổng kết ----------
print("\n" + "=" * 70)
print("  ✅ HOÀN TẤT THÍ NGHIỆM")
print("=" * 70)
print(f"  📁 Thư mục output: {OUTPUT_DIR}")
print(f"  📊 Bảng so sánh  : comparison_table.csv")
print(f"  📏 Phân tích size : size_analysis.csv")
print(f"  📈 Delta          : delta_vs_baseline.csv")
print(f"  📉 Biểu đồ       : experiments_comparison.png")
print(f"  📋 Raw JSON       : experiment_results.json")
print("=" * 70)

# Đề xuất thí nghiệm tốt nhất
best_exp = max(all_results.keys(), key=lambda k: all_results[k]["F1"])
best_f1 = all_results[best_exp]["F1"]
best_recall = all_results[best_exp]["Recall"]
print(f"\n  🏆 Thí nghiệm tốt nhất (theo F1): {best_exp}")
print(f"     F1 = {best_f1}, Recall = {best_recall}")

best_recall_exp = max(all_results.keys(), key=lambda k: all_results[k]["Recall"])
br = all_results[best_recall_exp]["Recall"]
print(f"\n  🎯 Recall cao nhất: {best_recall_exp}")
print(f"     Recall = {br}")
print(f"\n  💡 Trong pipeline 2-Stage, ưu tiên Recall cao (ít bỏ sót)")
print(f"     → nên dùng cấu hình của '{best_recall_exp}' cho Stage 1")
print("=" * 70)
