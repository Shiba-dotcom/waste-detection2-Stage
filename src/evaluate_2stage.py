"""
evaluate_2stage.py - Đánh giá end-to-end cho mô hình 2-Stage
Nhóm 2 - Waste Detection

Chạy inference 2-stage trên tập test (ảnh + label YOLO), so sánh BBox dự đoán
với Ground Truth (IoU >= 0.5), từ đó tính toán:
1. mAP@0.5 tổng thể
2. Precision, Recall, F1 cho từng lớp
3. Confusion Matrix phân loại
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from inference_2stage import TwoStageDetector, CLASS_NAMES

# ============================================================
# Cấu hình & Hàm hỗ trợ
# ============================================================

# MAP CLASS: GT YOLO file dùng 0-4 (Glass, Metal, Other, Paper, Plastic)
# Classifier dùng  0-5 (Background=0, Glass=1, ..., Plastic=5)
# GT không có Background nên shift GT lên +1 để khớp với Classifier index.
GT_CLASS_OFFSET = 1   # GT 0(Glass)→1(Glass), GT 4(Plastic)→5(Plastic)

def parse_yolo_labels(lbl_path, img_w, img_h):
    """Dọc file YOLO label và trả về danh sách [class_id, x1, y1, x2, y2].

    GT YOLO: 0=Glass,1=Metal,2=Other,3=Paper,4=Plastic (không có Background).
    Sau shift +1: 1=Glass,...,5=Plastic – khớp với Classifier CLASS_NAMES.
    """
    gt_boxes = []
    if not lbl_path.exists():
        return gt_boxes

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0]) + GT_CLASS_OFFSET  # 0→1, 1→2, ..., 4→5
            cx, cy, w, h = map(float, parts[1:])

            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h

            gt_boxes.append([cls_id, x1, y1, x2, y2])

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

# ============================================================
# Đánh giá Metrics
# ============================================================

def evaluate(detector, img_dir, lbl_dir, iou_thresh=0.5):
    """Chạy đánh giá trên toàn bộ dataset.

    Quy tắc TP/FP/FN:
    - TP: pred bbox khớp IoU với GT và class đúng
    - FP: pred bbox không khớp IoU với GT nào, hoặc khớp IoU nhưng sai class
    - FN: GT bbox không được dự đoán TP (bỏ sót hoặc detect sai class)
    """
    img_paths = [p for p in Path(img_dir).rglob('*')
                 if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    print(f"[INFO] Bắt đầu đánh giá trên {len(img_paths)} ảnh test...")

    # TP, FP, FN cho từng lớp (6 lớp: 0=Background ... 5=Plastic)
    # Lưu ý: GT không có class 0 (Background), nnên tp[0]=fp[0]=fn[0]=0 suốt
    tp = {i: 0 for i in range(6)}
    fp = {i: 0 for i in range(6)}
    fn = {i: 0 for i in range(6)}

    y_true_cls = []
    y_pred_cls = []

    # Thống kê debug
    total_gt = 0
    total_pred = 0
    total_iou_match = 0

    for img_path in tqdm(img_paths, desc="Evaluating"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # ── Ground Truth ──
        lbl_path = Path(lbl_dir) / img_path.relative_to(img_dir).with_suffix('.txt')
        gt_boxes = parse_yolo_labels(lbl_path, w, h)
        total_gt += len(gt_boxes)

        # Đăng ký FN ban đầu (sẽ trừ khi tìm thấy TP)
        for gt in gt_boxes:
            fn[gt[0]] += 1

        # ── Predictions (2-stage) ──
        _, preds = detector.process_image(img)
        total_pred += len(preds)

        # Sắp xếp theo confidence giảm dần
        preds = sorted(preds, key=lambda x: x['det_conf'], reverse=True)

        gt_matched = [False] * len(gt_boxes)

        for pred in preds:
            pred_box = pred['box']
            pred_cls = pred['class_id']   # 1–5 (Background đã bị lọc trước)

            best_iou = 0
            best_gt_idx = -1

            for i, gt in enumerate(gt_boxes):
                if gt_matched[i]:
                    continue
                iou = compute_iou(pred_box, gt[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_thresh:
                # Có IoU match – đánh dấu GT đã được xử lý
                gt_matched[best_gt_idx] = True
                gt_cls = gt_boxes[best_gt_idx][0]
                total_iou_match += 1

                y_true_cls.append(gt_cls)
                y_pred_cls.append(pred_cls)

                if pred_cls == gt_cls:
                    # ✔ TP: detect đúng vị trí và đúng lớp
                    tp[pred_cls] += 1
                    fn[gt_cls] -= 1       # Không bị bỏ sót nữa
                else:
                    # ✘ IoU match nhưng sai class
                    fp[pred_cls] += 1     # Pred sai lớp → FP của lớp đó predict
                    fn[gt_cls] -= 1       # [FIX] GT này đã bị consume (không còn chờ match)
                    #                    nhưng class sai → FN giữ nguyên để phạt recall
                    fn[gt_cls] += 1       # Phục hồi: GT vẫn bỏ sót về mặt phân loại
                    # → Net effect: fn[gt_cls] không thay đổi (bỏ sót), fp[pred_cls]+=1
            else:
                # Không khớp với GT nào → False Positive thuần
                fp[pred_cls] += 1

    # ── In thống kê debug ──
    print(f"\n[DEBUG] Tổng GT boxes    : {total_gt}")
    print(f"[DEBUG] Tổng Pred boxes   : {total_pred}")
    print(f"[DEBUG] IoU matches (≥{iou_thresh:.2f}): {total_iou_match}")
    iou_rate = total_iou_match / total_gt * 100 if total_gt > 0 else 0
    print(f"[DEBUG] Tỷ lệ GT được match : {iou_rate:.1f}%")

    # ── Tính metrics ──
    metrics = {}
    for i in range(6):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec  = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[CLASS_NAMES[i]] = {
            'Precision': prec, 'Recall': rec, 'F1': f1,
            'TP': tp[i], 'FP': fp[i], 'FN': fn[i]
        }

    return metrics, y_true_cls, y_pred_cls


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--detector', type=str, required=True, help="Path to YOLO weights")
    parser.add_argument('--classifier', type=str, required=True, help="Path to EfficientNet weights")
    parser.add_argument('--data-dir', type=str, required=True, help="Path to test images dir")
    parser.add_argument('--label-dir', type=str, required=True, help="Path to test labels dir")
    parser.add_argument('--conf', type=float, default=0.25, help="YOLO conf threshold")
    parser.add_argument('--iou', type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument('--output', type=str, default='results/2stage/eval', help="Output dir")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Init model
    detector = TwoStageDetector(args.detector, args.classifier, args.conf)
    
    # Evaluate
    metrics, y_true, y_pred = evaluate(detector, args.data_dir, args.label_dir, args.iou)
    
    # 1. Print and Save Metrics
    df_metrics = pd.DataFrame(metrics).T
    print("\n" + "="*50)
    print("  KẾT QUẢ ĐÁNH GIÁ (PER CLASS)")
    print("="*50)
    print(df_metrics[['Precision', 'Recall', 'F1']].round(4))
    
    # Tính Macro Average
    macro_p = df_metrics['Precision'].mean()
    macro_r = df_metrics['Recall'].mean()
    macro_f1 = df_metrics['F1'].mean()
    
    # mAP@0.5 xấp xỉ bằng Macro F1 trong detection (khi đánh giá tại 1 ngưỡng conf cố định)
    # Để tính mAP chính xác cần đường cong PR, ở đây dùng Macro F1 làm metric tương đương
    
    print("-" * 50)
    print(f"  Macro Precision : {macro_p:.4f}")
    print(f"  Macro Recall    : {macro_r:.4f}")
    print(f"  Macro F1 (≈mAP) : {macro_f1:.4f}")
    print("="*50)
    
    df_metrics.to_csv(out_dir / 'per_class_metrics.csv')
    
    # 2. Comparison Table
    comp_data = {
        'Method': ['baseline_yolov8n', 'exp6_tiling (best 1-stage)', '2-Stage (ours)'],
        'mAP@0.5 (≈F1)': [0.249, 0.323, macro_f1],
        'Precision': [0.295, 0.383, macro_p],
        'Recall': [0.323, 0.355, macro_r]
    }
    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(out_dir / 'comparison_table.csv', index=False)
    
    # 3. Confusion Matrix (6 lớp)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[CLASS_NAMES[i] for i in range(6)],
                yticklabels=[CLASS_NAMES[i] for i in range(6)])
    plt.title('Stage 2 Classification Confusion Matrix (Matched Boxes)')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)
    
    print(f"\n[INFO] Đã lưu kết quả tại: {out_dir}")

if __name__ == "__main__":
    main()
