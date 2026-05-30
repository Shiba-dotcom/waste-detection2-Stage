import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import timm

# Thêm thư viện SAHI
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    print("[ERROR] Cần cài đặt SAHI: pip install sahi")
    sys.exit(1)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    u = a1 + a2 - inter
    return inter / u if u > 0 else 0.0

def parse_labels(lbl_path, w, h):
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cls += 1  # Shift GT (0-4) lên +1 để khớp Classifier (0=BG, 1-5=rác)
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                boxes.append({'class': cls, 'bbox': [x1, y1, x2, y2]})
    return boxes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", required=True, help="Path to YOLO pt file")
    parser.add_argument("--classifier", required=True, help="Path to Stage 2 pth file")
    parser.add_argument("--data-dir", required=True, help="Test images dir")
    parser.add_argument("--label-dir", required=True, help="Test multi-class labels dir")
    parser.add_argument("--conf", type=float, default=0.25, help="SAHI confidence threshold")
    parser.add_argument("--slice-size", type=int, default=512, help="SAHI slice size")
    parser.add_argument("--overlap", type=float, default=0.2, help="SAHI overlap ratio")
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[INFO] Khởi tạo SAHI AutoDetectionModel...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=args.detector,
        confidence_threshold=args.conf,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    print("[INFO] Khởi tạo EfficientNet-B2 (Stage 2)...")
    checkpoint = torch.load(args.classifier, map_location=device, weights_only=False)
    
    # Đọc thông tin từ checkpoint
    if "num_classes" in checkpoint:
        num_classes = checkpoint["num_classes"]
        classes = checkpoint.get("class_names", [f"Class_{i}" for i in range(num_classes)])
    else:
        num_classes = 6
        classes = ['Background', 'Glass', 'Metal', 'Other', 'Paper', 'Plastic']
    
    print(f"[INFO] num_classes = {num_classes}, classes = {classes}")
    
    # Tìm index của class Background (nếu có)
    bg_idx = classes.index("Background") if "Background" in classes else -1
    
    classifier = timm.create_model('efficientnet_b2', pretrained=False, num_classes=num_classes)
    in_features = classifier.classifier.in_features
    classifier.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(in_features, num_classes)
    )
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
    else:
        classifier.load_state_dict(checkpoint)
    classifier.to(device)
    classifier.eval()
    
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_dir = Path(args.data_dir)
    lbl_dir = Path(args.label_dir)
    
    img_paths = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))
    
    total_gt = 0
    total_pred = 0
    matched = 0
    
    # Ma trận nhầm lẫn
    tp = {c: 0 for c in range(num_classes)}
    fp = {c: 0 for c in range(num_classes)}
    fn = {c: 0 for c in range(num_classes)}
    
    print(f"[INFO] Bắt đầu SAHI Inference trên {len(img_paths)} ảnh...")
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        lbl_path = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
        gt_boxes = parse_labels(lbl_path, w, h)
        total_gt += len(gt_boxes)
        
        # 1. Chạy SAHI Sliced Prediction
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap
        )
        
        preds = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0: continue
            
            # 2. Phân loại qua Stage 2
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = img_transform(crop_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                out = classifier(crop_tensor)
                cls_id = out.argmax(1).item()
                
            if cls_id != bg_idx: # Bỏ qua Background (nếu có)
                preds.append({'class': cls_id, 'bbox': [x1, y1, x2, y2]})
                
        total_pred += len(preds)
        
        # Đối chiếu GT và Pred
        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(preds)
        
        for pi, p in enumerate(preds):
            best_iou = 0
            best_gi = -1
            for gi, g in enumerate(gt_boxes):
                if gt_matched[gi]: continue
                v = iou(p['bbox'], g['bbox'])
                if v > best_iou:
                    best_iou = v
                    best_gi = gi
            
            if best_iou >= 0.5:
                matched += 1
                gt_matched[best_gi] = True
                pred_matched[pi] = True
                if p['class'] == gt_boxes[best_gi]['class']:
                    tp[p['class']] += 1
                else:
                    fp[p['class']] += 1
                    fn[gt_boxes[best_gi]['class']] += 1
            else:
                fp[p['class']] += 1
                
        for gi, g in enumerate(gt_boxes):
            if not gt_matched[gi]:
                fn[g['class']] += 1
                
    print("\n" + "="*60)
    print("  KẾT QUẢ ĐÁNH GIÁ (SAHI + EFFICIENTNET)")
    print("="*60)
    print(f"Tổng GT boxes   : {total_gt}")
    print(f"Tổng Pred boxes : {total_pred}")
    print(f"IoU matches     : {matched} ({matched/max(total_gt,1)*100:.1f}%)")
    print("-"*60)
    print(f"{'Class':<12} {'Precision':>9} {'Recall':>9} {'F1':>9} {'TP':>5} {'FP':>5} {'FN':>5}")
    
    macro_p, macro_r, macro_f1 = 0, 0, 0
    count = 0
    
    for c in range(num_classes):
        if c == bg_idx: continue  # Bỏ qua Background
        t = tp[c]
        f_p = fp[c]
        f_n = fn[c]
        
        p = t / (t + f_p) if (t + f_p) > 0 else 0
        r = t / (t + f_n) if (t + f_n) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        cls_name = classes[c] if c < len(classes) else f"Class_{c}"
        print(f"{cls_name:<12} {p:>9.4f} {r:>9.4f} {f1:>9.4f} {t:>5} {f_p:>5} {f_n:>5}")
        macro_p += p
        macro_r += r
        macro_f1 += f1
        count += 1
        
    print("-" * 60)
    if count > 0:
        print(f"Macro P : {macro_p/count:.4f}")
        print(f"Macro R : {macro_r/count:.4f}")
        print(f"Macro F1: {macro_f1/count:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
