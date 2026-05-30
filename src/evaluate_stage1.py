"""
evaluate_stage1.py — Đánh giá riêng Stage 1 (YOLO Binary Detector)

Dùng data/processed_binary (ảnh + label binary: chỉ 1 class = Waste) để
đánh giá chất lượng thực sự của YOLO:
  - Bao nhiêu % rác được phát hiện (Recall)?
  - Bao nhiêu phần trăm detection là đúng (Precision)?
  - mAP@0.5 của binary detection

Chạy:
  python src/evaluate_stage1.py \
    --detector stage1_best.pt \
    --data-dir data/processed_binary/images/test \
    --label-dir data/processed_binary/labels/test \
    --conf 0.05
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm


# ============================================================
# Hàm hỗ trợ
# ============================================================

def parse_binary_labels(lbl_path, img_w, img_h):
    """Đọc YOLO binary label (class 0 = Waste).
    Trả về list [x1, y1, x2, y2] (pixel coords).
    """
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def compute_iou(box1, box2):
    """IoU giữa 2 boxes [x1,y1,x2,y2]."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# Đánh giá binary detection
# ============================================================

def evaluate_stage1(detector, img_dir, lbl_dir, conf_thresh=0.05, iou_thresh=0.5):
    """Đánh giá YOLO binary detector.

    Returns:
        dict với các metric và danh sách per-image stats.
    """
    from ultralytics import YOLO

    model = YOLO(detector)

    img_paths = sorted([
        p for p in Path(img_dir).rglob('*')
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ])
    print(f"[INFO] Số ảnh test: {len(img_paths)}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt  = 0
    total_det = 0

    # Lưu để vẽ biểu đồ phân phối IoU
    all_max_ious = []       # IoU cao nhất cho mỗi GT box
    conf_tp = []            # confidence của TP detections
    conf_fp = []            # confidence của FP detections

    for img_path in tqdm(img_paths, desc="Stage 1 Eval"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Ground Truth
        lbl_path = Path(lbl_dir) / img_path.relative_to(img_dir).with_suffix('.txt')
        gt_boxes = parse_binary_labels(lbl_path, w, h)
        total_gt += len(gt_boxes)

        # Prediction
        results = model(img_path, conf=conf_thresh, verbose=False)[0]
        pred_boxes = results.boxes.xyxy.cpu().numpy().tolist()
        pred_confs = results.boxes.conf.cpu().numpy().tolist()
        total_det += len(pred_boxes)

        # Matching: mỗi GT chỉ match với 1 pred (greedy, theo conf giảm dần)
        gt_matched  = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # Sắp xếp pred theo conf giảm dần
        order = sorted(range(len(pred_boxes)), key=lambda i: pred_confs[i], reverse=True)

        for pi in order:
            pb = pred_boxes[pi]
            best_iou = 0
            best_gi  = -1
            for gi, gb in enumerate(gt_boxes):
                if gt_matched[gi]:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi

            if best_iou >= iou_thresh:
                total_tp += 1
                gt_matched[best_gi]   = True
                pred_matched[pi]      = True
                conf_tp.append(pred_confs[pi])
            else:
                total_fp += 1
                conf_fp.append(pred_confs[pi])

        total_fn += gt_matched.count(False)

        # Lưu max IoU cho từng GT (dù có match hay không)
        for gb in gt_boxes:
            if pred_boxes:
                ious = [compute_iou(pb, gb) for pb in pred_boxes]
                all_max_ious.append(max(ious))
            else:
                all_max_ious.append(0.0)

    # ── Metrics ──
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'total_gt':    total_gt,
        'total_det':   total_det,
        'TP': total_tp, 'FP': total_fp, 'FN': total_fn,
        'Precision': precision,
        'Recall':    recall,
        'F1':        f1,
        'conf_tp':   conf_tp,
        'conf_fp':   conf_fp,
        'all_max_ious': all_max_ious,
    }


# ============================================================
# Vẽ biểu đồ
# ============================================================

def plot_results(stats, out_dir, conf_thresh):
    """Vẽ 3 biểu đồ: bar metric, IoU distribution, Confidence distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Stage 1 (YOLO Binary Detector) — conf={conf_thresh}", fontsize=13, fontweight='bold')

    # 1. Bar chart: Precision / Recall / F1
    ax = axes[0]
    names  = ['Precision', 'Recall', 'F1']
    values = [stats['Precision'], stats['Recall'], stats['F1']]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_title('Binary Detection Metrics')
    ax.set_ylabel('Score')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.98, 0.95,
            f"GT: {stats['total_gt']}\nDet: {stats['total_det']}\n"
            f"TP: {stats['TP']}  FP: {stats['FP']}  FN: {stats['FN']}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', alpha=0.1))

    # 2. Histogram: Phân phối max-IoU của GT boxes
    ax = axes[1]
    ious = stats['all_max_ious']
    ax.hist(ious, bins=20, range=(0, 1), color='#5C6BC0', edgecolor='white', linewidth=0.8)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='IoU threshold=0.5')
    matched_pct = sum(1 for v in ious if v >= 0.5) / len(ious) * 100 if ious else 0
    ax.set_title(f'Max-IoU Distribution per GT Box\n(GT bị match: {matched_pct:.1f}%)')
    ax.set_xlabel('Max IoU với pred bbox')
    ax.set_ylabel('Số GT boxes')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. Confidence distribution: TP vs FP
    ax = axes[2]
    if stats['conf_tp']:
        ax.hist(stats['conf_tp'], bins=20, range=(0, 1), alpha=0.7,
                color='#43A047', edgecolor='white', label=f"TP (n={len(stats['conf_tp'])})")
    if stats['conf_fp']:
        ax.hist(stats['conf_fp'], bins=20, range=(0, 1), alpha=0.7,
                color='#E53935', edgecolor='white', label=f"FP (n={len(stats['conf_fp'])})")
    ax.axvline(x=conf_thresh, color='black', linestyle=':', linewidth=1.5,
               label=f'conf={conf_thresh}')
    ax.set_title('Confidence Distribution: TP vs FP')
    ax.set_xlabel('Detection Confidence')
    ax.set_ylabel('Count')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / 'stage1_eval.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"[INFO] Đã lưu biểu đồ: {out_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Đánh giá riêng Stage 1 – YOLO Binary Detector")
    parser.add_argument('--detector',  required=True,  help="Path to YOLO weights (stage1_best.pt)")
    parser.add_argument('--data-dir',  required=True,  help="Thư mục ảnh test (processed_binary/images/test)")
    parser.add_argument('--label-dir', required=True,  help="Thư mục label binary (processed_binary/labels/test)")
    parser.add_argument('--conf',      type=float, default=0.05, help="Confidence threshold (default: 0.05)")
    parser.add_argument('--iou',       type=float, default=0.50, help="IoU threshold (default: 0.5)")
    parser.add_argument('--output',    default='results/stage1_eval', help="Thư mục lưu kết quả")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  ĐÁNH GIÁ STAGE 1 – YOLO BINARY DETECTOR")
    print("=" * 60)
    print(f"  Detector : {args.detector}")
    print(f"  Data     : {args.data_dir}")
    print(f"  Labels   : {args.label_dir}")
    print(f"  conf     : {args.conf}   iou: {args.iou}")
    print("=" * 60)

    stats = evaluate_stage1(
        detector   = args.detector,
        img_dir    = args.data_dir,
        lbl_dir    = args.label_dir,
        conf_thresh= args.conf,
        iou_thresh = args.iou,
    )

    # ── In kết quả ──
    print("\n" + "=" * 60)
    print("  KẾT QUẢ STAGE 1 (Binary Detection)")
    print("=" * 60)
    print(f"  Tổng GT boxes      : {stats['total_gt']:,}")
    print(f"  Tổng Detections    : {stats['total_det']:,}")
    print(f"  TP                 : {stats['TP']:,}")
    print(f"  FP                 : {stats['FP']:,}")
    print(f"  FN                 : {stats['FN']:,}")
    print("-" * 60)
    print(f"  Precision          : {stats['Precision']:.4f}  ({stats['Precision']*100:.1f}%)")
    print(f"  Recall             : {stats['Recall']:.4f}  ({stats['Recall']*100:.1f}%)")
    print(f"  F1 Score           : {stats['F1']:.4f}  ({stats['F1']*100:.1f}%)")

    # % GT được detect (max IoU >= 0.5)
    ious = stats['all_max_ious']
    pct_matched = sum(1 for v in ious if v >= args.iou) / len(ious) * 100 if ious else 0
    print(f"  % GT bị phát hiện  : {pct_matched:.1f}%  (IoU>={args.iou})")
    print("=" * 60)

    # ── Lưu CSV ──
    df = pd.DataFrame([{
        'conf_thresh': args.conf,
        'iou_thresh':  args.iou,
        'Total_GT':    stats['total_gt'],
        'Total_Det':   stats['total_det'],
        'TP': stats['TP'], 'FP': stats['FP'], 'FN': stats['FN'],
        'Precision':   round(stats['Precision'], 4),
        'Recall':      round(stats['Recall'],    4),
        'F1':          round(stats['F1'],        4),
        'GT_matched_%': round(pct_matched, 2),
    }])
    csv_path = out_dir / 'stage1_metrics.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"[INFO] Đã lưu metrics: {csv_path}")

    # ── Vẽ biểu đồ ──
    plot_results(stats, out_dir, args.conf)


if __name__ == '__main__':
    main()
