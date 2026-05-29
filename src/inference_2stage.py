"""
inference_2stage.py - Pipeline kết hợp YOLO (Stage 1) và EfficientNet (Stage 2)
Nhóm 2 - Waste Detection

Sử dụng:
  - Stage 1: YOLO phát hiện các bounding box của "Waste".
  - Crop: Cắt các bounding box từ ảnh gốc.
  - Stage 2: EfficientNet phân loại các crop thành 5 lớp rác.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import timm
from pathlib import Path
import argparse
import time
from ultralytics import YOLO

# ============================================================
# Cấu hình & Hàm hỗ trợ
# ============================================================

# Class mapping của dự án (thêm Background)
CLASS_NAMES = {0: 'Background', 1: 'Glass', 2: 'Metal', 3: 'Other', 4: 'Paper', 5: 'Plastic'}

# Bảng màu (BGR cho OpenCV)
COLORS = {
    'Background': (128, 128, 128), # Gray (nếu muốn debug)
    'Glass': (54, 67, 244),   # Red
    'Metal': (243, 150, 33),  # Blue
    'Other': (0, 152, 255),   # Orange
    'Paper': (176, 39, 156),  # Purple
    'Plastic': (80, 175, 76), # Green
}

def load_classifier(weights_path, device):
    """Khởi tạo và load trọng số cho model Classification (EfficientNet-B2).
    
    Tái tạo đúng kiến trúc head Sequential(Dropout, Linear) như khi train.
    Tự động đọc num_classes và class_names từ checkpoint.
    """
    # ── Bước 1: Đọc checkpoint trước để lấy thông tin cấu trúc ──
    num_classes = 6   # fallback mặc định
    dropout_rate = 0.4
    state_dict = None

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device)

        # Đọc metadata từ checkpoint (nếu có)
        if isinstance(checkpoint, dict):
            num_classes  = checkpoint.get('num_classes',  num_classes)
            state_dict   = checkpoint.get('model_state_dict', checkpoint)
            loaded_names = checkpoint.get('class_names', None)
            if loaded_names:
                # Cập nhật CLASS_NAMES toàn cục theo checkpoint
                global CLASS_NAMES
                CLASS_NAMES = {i: name for i, name in enumerate(loaded_names)}
        else:
            state_dict = checkpoint

        print(f"[INFO] Đã load classifier weights từ: {weights_path}")
        print(f"[INFO] num_classes = {num_classes}, classes = {list(CLASS_NAMES.values())}")
    else:
        print(f"[WARN] Không tìm thấy classifier weights tại {weights_path}, sử dụng weights khởi tạo!")

    # ── Bước 2: Tạo backbone (không custom head) ──
    model = timm.create_model('efficientnet_b2', pretrained=False)

    # ── Bước 3: Gắn đúng head Sequential(Dropout, Linear) như lúc train ──
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=dropout_rate),
        torch.nn.Linear(in_features, num_classes)
    )

    # ── Bước 4: Load weights ──
    if state_dict is not None:
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

def get_transforms():
    """Transform cho ảnh trước khi đưa vào classifier (giống validation/test)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def crop_bbox(img, x1, y1, x2, y2, pad_ratio=0.10):
    """Cắt bounding box từ ảnh gốc có kèm padding."""
    h, w = img.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    
    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)
    
    cx1 = max(0, int(x1 - pad_x))
    cy1 = max(0, int(y1 - pad_y))
    cx2 = min(w, int(x2 + pad_x))
    cy2 = min(h, int(y2 + pad_y))
    
    crop = img[cy1:cy2, cx1:cx2]
    return crop

def draw_prediction(img, box, cls_name, conf_det, conf_cls):
    """Vẽ bounding box và nhãn lên ảnh."""
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(cls_name, (255, 255, 255))
    
    # Vẽ bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Label text: "Class XX% (Det: YY%)"
    label = f"{cls_name} {conf_cls:.0%} (Det: {conf_det:.0%})"
    
    # Kích thước text background
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - baseline - 5), (x1 + tw, y1), color, -1)
    
    # Vẽ text
    cv2.putText(img, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

# ============================================================
# Core Pipeline
# ============================================================

class TwoStageDetector:
    def __init__(self, detector_path, classifier_path, conf_thresh=0.25):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Đang sử dụng thiết bị: {self.device}")
        
        # Load Stage 1 (YOLO)
        self.detector = YOLO(detector_path)
        self.conf_thresh = conf_thresh
        
        # Load Stage 2 (EfficientNet)
        self.classifier = load_classifier(Path(classifier_path), self.device)
        self.transform = get_transforms()
        
    def process_image(self, img_bgr):
        """Xử lý 1 ảnh BGR qua 2 stage."""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result_img = img_bgr.copy()
        
        # --- Stage 1: Detection ---
        det_results = self.detector(img_bgr, conf=self.conf_thresh, verbose=False)[0]
        boxes = det_results.boxes.xyxy.cpu().numpy()
        det_confs = det_results.boxes.conf.cpu().numpy()
        
        predictions = []
        
        # --- Stage 2: Classification ---
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            
            # Cắt ảnh RGB
            crop = crop_bbox(img_rgb, x1, y1, x2, y2)
            if crop is None or crop.size == 0:
                continue
                
            # Chuẩn bị input cho classifier
            input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.classifier(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                cls_idx = torch.argmax(probs).item()
                cls_conf = probs[cls_idx].item()
                
            cls_name = CLASS_NAMES[cls_idx]
            
            # --- [MỚI] Lọc False Positives ---
            # Nếu model phân loại đây là Background (không phải rác),
            # chúng ta sẽ âm thầm bỏ qua bounding box này.
            if cls_name == 'Background':
                continue
                
            predictions.append({
                'box': box,
                'class_id': cls_idx,
                'class_name': cls_name,
                'det_conf': det_confs[i],
                'cls_conf': cls_conf
            })
            
            # Vẽ kết quả
            result_img = draw_prediction(result_img, box, cls_name, det_confs[i], cls_conf)
            
        return result_img, predictions

# ============================================================
# Main Execution
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="2-Stage Inference Pipeline")
    parser.add_argument('--image', type=str, help="Đường dẫn đến 1 ảnh")
    parser.add_argument('--dir', type=str, help="Đường dẫn đến thư mục chứa ảnh")
    parser.add_argument('--webcam', action='store_true', help="Sử dụng webcam")
    parser.add_argument('--detector', type=str, required=True, help="Đường dẫn trọng số YOLO (Stage 1)")
    parser.add_argument('--classifier', type=str, required=True, help="Đường dẫn trọng số EfficientNet (Stage 2)")
    parser.add_argument('--conf', type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument('--output', type=str, default='results/2stage/inference', help="Thư mục lưu kết quả")
    parser.add_argument('--no-display', action='store_true', help="Không hiển thị ảnh (hữu ích trên server/Kaggle)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo detector
    pipeline = TwoStageDetector(args.detector, args.classifier, args.conf)
    
    # 1. Chế độ 1 ảnh
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"[LỖI] Không tìm thấy ảnh: {img_path}")
            return
            
        print(f"[INFO] Xử lý ảnh: {img_path}")
        img = cv2.imread(str(img_path))
        
        t0 = time.time()
        res_img, preds = pipeline.process_image(img)
        t1 = time.time()
        
        print(f"[INFO] Xử lý xong trong {(t1-t0)*1000:.1f}ms. Phát hiện {len(preds)} vật thể.")
        
        out_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), res_img)
        print(f"[INFO] Đã lưu kết quả tại: {out_path}")
        
        if not args.no_display:
            cv2.imshow("Result", res_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    # 2. Chế độ thư mục
    elif args.dir:
        in_dir = Path(args.dir)
        if not in_dir.exists():
            print(f"[LỖI] Không tìm thấy thư mục: {in_dir}")
            return
            
        img_paths = [p for p in in_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"[INFO] Tìm thấy {len(img_paths)} ảnh trong {in_dir}")
        
        total_time = 0
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: continue
            
            t0 = time.time()
            res_img, _ = pipeline.process_image(img)
            total_time += (time.time() - t0)
            
            out_path = out_dir / f"pred_{p.name}"
            cv2.imwrite(str(out_path), res_img)
            
        if len(img_paths) > 0:
            avg_time = total_time / len(img_paths)
            print(f"[INFO] Hoàn tất! Thời gian trung bình: {avg_time*1000:.1f}ms/ảnh")
            print(f"[INFO] Kết quả lưu tại: {out_dir}")
            
    # 3. Chế độ Webcam
    elif args.webcam:
        print("[INFO] Đang mở webcam... (Nhấn 'q' để thoát)")
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            res_img, _ = pipeline.process_image(frame)
            
            if not args.no_display:
                cv2.imshow("Waste Detection 2-Stage", res_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        print("[WARN] Vui lòng chọn nguồn dữ liệu: --image, --dir, hoặc --webcam")
        parser.print_help()

if __name__ == "__main__":
    main()
