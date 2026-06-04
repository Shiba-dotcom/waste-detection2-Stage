# -*- coding: utf-8 -*-
"""
Demo GUI - Waste Detection & Classification
4 chế độ phân tích:
  1. 2-Stage Best    : 2-Stage_best.pt + 2-Stage_best.pth
  2. 2-Stage SAHI    : 2-Stage_SAHI.pt + 2-Stage_SAHI.pth
  3. 1-Stage Baseline: 1_StageBaseline.pt  (phát hiện + phân loại 1 bước)
  4. 1-Stage SAHI    : 1_StageSAHI.pt      (phát hiện + phân loại 1 bước)

Giao diện với 3 chế độ nhập:
  1. Upload hình ảnh
  2. Chụp ảnh từ webcam
  3. Nhận diện video trực tiếp (Live Camera)

Lưu ý: Lớp "Background" chỉ dùng để lọc — không hiển thị như lớp rác.
"""

import os
import sys
import threading
import time

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    HAS_SAHI = True
except ImportError:
    HAS_SAHI = False

# ══════════════════════════════════════════════════════════════
# 0. CẤU HÌNH MẶC ĐỊNH
# ══════════════════════════════════════════════════════════════

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 chế độ phân tích
ANALYSIS_MODES = {
    1: {
        "name":     "2-Stage Best",
        "desc":     "YOLO phát hiện → EfficientNet phân loại",
        "pt":       os.path.join(_BASE, "models", "2-Stage_best.pt"),
        "pth":      os.path.join(_BASE, "models", "2-Stage_best.pth"),
        "one_stage": False,
        "use_sahi": False,
    },
    2: {
        "name":     "2-Stage SAHI",
        "desc":     "YOLO-SAHI phát hiện → EfficientNet phân loại",
        "pt":       os.path.join(_BASE, "models", "2-Stage_SAHI.pt"),
        "pth":      os.path.join(_BASE, "models", "2-Stage_SAHI.pth"),
        "one_stage": False,
        "use_sahi": True,
    },
    3: {
        "name":     "1-Stage Baseline",
        "desc":     "YOLO phát hiện & phân loại 1 bước",
        "pt":       os.path.join(_BASE, "models", "1_StageBaseline.pt"),
        "pth":      None,
        "one_stage": True,
        "use_sahi": False,
    },
    4: {
        "name":     "1-Stage SAHI",
        "desc":     "YOLO-SAHI phát hiện & phân loại 1 bước",
        "pt":       os.path.join(_BASE, "models", "1_StageSAHI.pt"),
        "pth":      None,
        "one_stage": True,
        "use_sahi": True,
    },
}

OUTPUT_DIR = os.path.join(_BASE, "results")

# ── Theme: Eco / Waste Classification ──
BG_MAIN       = "#0c1210"
BG_SURFACE    = "#141f1a"
BG_CARD       = "#1a2822"
BG_ELEVATED   = "#223329"
BG_INPUT      = "#0a100e"

PRIMARY       = "#22c55e"
PRIMARY_DARK  = "#16a34a"
PRIMARY_LIGHT = "#4ade80"
ACCENT        = "#2dd4bf"
ACCENT2       = "#0d9488"
WARNING       = "#f59e0b"
DANGER        = "#ef4444"
TEXT_PRIMARY  = "#ecfdf5"
TEXT_SECONDARY= "#cbd5e1"
TEXT_MUTED    = "#64748b"
BORDER        = "#2a3f35"
BORDER_LIGHT  = "#3d5248"

BG_DARK  = BG_MAIN
BG_PANEL = BG_ELEVATED
GREEN    = PRIMARY
ORANGE   = WARNING
RED_SOFT = DANGER

FONT_FAMILY = "Segoe UI"
FONT_TITLE  = (FONT_FAMILY, 20, "bold")
FONT_HEAD   = (FONT_FAMILY, 11, "bold")
FONT_BODY   = (FONT_FAMILY, 10)
FONT_SMALL  = (FONT_FAMILY, 9)
FONT_CAPTION= (FONT_FAMILY, 8)

# Lớp Background chỉ dùng để lọc, KHÔNG phân loại rác
BACKGROUND_CLASS = "Background"

# 6 lớp EfficientNet (đúng thứ tự checkpoint)
CLASSIFIER_CLASSES = [
    "Background", "Glass", "Metal", "Other", "Paper", "Plastic",
]
# Chỉ 5 lớp rác thực sự (không kể Background)
WASTE_CLASSES = [c for c in CLASSIFIER_CLASSES if c != BACKGROUND_CLASS]

CLASS_META = {
    "Background": {"icon": "◻", "vi": "Nền / không phải rác"},
    "Glass":      {"icon": "◆", "vi": "Thủy tinh"},
    "Metal":      {"icon": "▣", "vi": "Kim loại"},
    "Other":      {"icon": "●", "vi": "Rác khác"},
    "Paper":      {"icon": "▤", "vi": "Giấy"},
    "Plastic":    {"icon": "▲", "vi": "Nhựa"},
    "Waste":      {"icon": "♻", "vi": "Vùng rác (YOLO)"},
}

CLASS_COLORS = {
    "Background": "#64748b",
    "Glass":      "#38bdf8",
    "Metal":      "#fbbf24",
    "Other":      "#a78bfa",
    "Paper":      "#34d399",
    "Plastic":    "#f87171",
    "Waste":      "#fb923c",
}

def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ══════════════════════════════════════════════════════════════
# 1. MODEL LOADING
# ══════════════════════════════════════════════════════════════

class SimpleClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def load_cls_model(pth_path: str):
    """Load EfficientNet classification model từ .pth checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(pth_path, map_location=device, weights_only=False)
    class_names = None
    img_size = 224
    num_classes = 10

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict",
                     checkpoint.get("model",
                     checkpoint.get("state_dict", checkpoint)))
        class_names = checkpoint.get("class_names", None)
        img_size    = checkpoint.get("img_size", 224)
        num_classes = checkpoint.get("num_classes", num_classes)

        is_timm_effnet = any("conv_stem" in k for k in state_dict.keys())
        if is_timm_effnet:
            import timm
            model = timm.create_model("efficientnet_b2", pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(model.classifier.in_features, num_classes)
            )
        else:
            model = SimpleClassifier(num_classes=num_classes)

        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Không nhận ra định dạng checkpoint: {type(checkpoint)}")

    model.to(device)
    model.eval()
    return model, device, class_names, img_size


def load_yolo_model(pt_path: str):
    """Load YOLO detection model từ .pt file (ultralytics)."""
    from ultralytics import YOLO
    model = YOLO(pt_path)
    return model


# ══════════════════════════════════════════════════════════════
# 2. INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════

def build_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def predict_cls(cls_model, device, pil_img: Image.Image,
                image_size: int, class_names, top_k: int = 5):
    """Phân loại ảnh với EfficientNet, trả về top-k results."""
    transform = build_transform(image_size)
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cls_model(tensor)
        probs  = torch.softmax(logits, dim=1)
    top_probs, top_idxs = probs.topk(min(top_k, probs.shape[1]), dim=1)
    top_probs = top_probs.squeeze().cpu().tolist()
    top_idxs  = top_idxs.squeeze().cpu().tolist()
    if not isinstance(top_probs, list):
        top_probs = [top_probs]
        top_idxs  = [top_idxs]
    results = []
    for prob, idx in zip(top_probs, top_idxs):
        label = (class_names[idx] if class_names and idx < len(class_names)
                 else f"class_{idx}")
        results.append({"mode": "classify", "class_id": idx,
                         "label": label, "confidence": round(float(prob), 4)})
    return results


def predict_with_detection(yolo_model, cls_model, device,
                            pil_img: Image.Image, img_size: int,
                            class_names, conf_thresh: float = 0.40):
    """
    Pipeline 2 bước:
      1. YOLO → detect bbox vùng rác
      2. EfficientNet → classify từng crop bbox → lấy top-1
    Lớp Background bị lọc ra khỏi kết quả hiển thị.
    """
    import numpy as np
    img_np = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    W, H = pil_img.size
    detections = []

    results_yolo = yolo_model(img_np, conf=conf_thresh, verbose=False)
    for r in results_yolo:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            xyxy   = box.xyxy[0].cpu().tolist()
            conf_d = float(box.conf[0].cpu())

            x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
            x2 = min(x2, W)
            y2 = min(y2, H)

            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            # Bước 2: crop và classify
            crop = pil_img.crop((x1, y1, x2, y2))
            cls_result = predict_cls(cls_model, device, crop,
                                     img_size, class_names, top_k=1)
            top_label = cls_result[0]["label"] if cls_result else "Waste"
            top_conf  = cls_result[0]["confidence"] if cls_result else conf_d

            # Lọc Background — không hiển thị như lớp rác
            if top_label == BACKGROUND_CLASS:
                continue

            detections.append({
                "mode":       "detect",
                "bbox":       [x1, y1, x2, y2],
                "conf_det":   round(conf_d, 3),
                "label":      top_label,
                "confidence": round(top_conf, 4),
            })

    if not detections:
        cls_res = predict_cls(cls_model, device, pil_img,
                              img_size, class_names, top_k=5)
        # Lọc Background khỏi kết quả classify fallback
        cls_res = [r for r in cls_res if r["label"] != BACKGROUND_CLASS]
        return cls_res, False

    return detections, True


def predict_one_stage(yolo_model, pil_img: Image.Image,
                      conf_thresh: float = 0.40):
    """
    Pipeline 1 bước: YOLO vừa phát hiện vừa phân loại.
    Lớp Background bị lọc ra.
    """
    import numpy as np
    img_np = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    W, H = pil_img.size
    detections = []

    results_yolo = yolo_model(img_np, conf=conf_thresh, verbose=False)
    for r in results_yolo:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        names = r.names  # dict {id: class_name}
        for box in boxes:
            xyxy   = box.xyxy[0].cpu().tolist()
            conf_d = float(box.conf[0].cpu())
            cls_id = int(box.cls[0].cpu())
            label  = names.get(cls_id, f"class_{cls_id}")

            # Lọc Background
            if label == BACKGROUND_CLASS:
                continue

            x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
            x2 = min(x2, W)
            y2 = min(y2, H)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            detections.append({
                "mode":       "detect",
                "bbox":       [x1, y1, x2, y2],
                "conf_det":   round(conf_d, 3),
                "label":      label,
                "confidence": round(conf_d, 4),
            })

    return detections, bool(detections)


def predict_with_detection_sahi(sahi_model, cls_model, device,
                                pil_img: Image.Image, img_size: int,
                                class_names):
    """Pipeline 2 bước sử dụng SAHI cho khâu phát hiện (YOLO)."""
    import numpy as np
    W, H = pil_img.size
    detections = []
    
    # SAHI inference (cắt ảnh thành các miếng 640x640, overlap 20%)
    result = get_sliced_prediction(
        pil_img,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )
    
    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_xyxy()
        conf_d = obj.score.value
        
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2 = min(x2, W)
        y2 = min(y2, H)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
            
        crop = pil_img.crop((x1, y1, x2, y2))
        cls_result = predict_cls(cls_model, device, crop,
                                 img_size, class_names, top_k=1)
        top_label = cls_result[0]["label"] if cls_result else "Waste"
        top_conf  = cls_result[0]["confidence"] if cls_result else conf_d

        if top_label == BACKGROUND_CLASS:
            continue

        detections.append({
            "mode":       "detect",
            "bbox":       [x1, y1, x2, y2],
            "conf_det":   round(conf_d, 3),
            "label":      top_label,
            "confidence": round(top_conf, 4),
        })

    if not detections:
        cls_res = predict_cls(cls_model, device, pil_img,
                              img_size, class_names, top_k=5)
        cls_res = [r for r in cls_res if r["label"] != BACKGROUND_CLASS]
        return cls_res, False

    return detections, True


def predict_one_stage_sahi(sahi_model, pil_img: Image.Image):
    """Pipeline 1 bước sử dụng SAHI."""
    W, H = pil_img.size
    detections = []

    result = get_sliced_prediction(
        pil_img,
        sahi_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=0
    )

    for obj in result.object_prediction_list:
        bbox = obj.bbox.to_xyxy()
        conf_d = obj.score.value
        label = obj.category.name

        if label == BACKGROUND_CLASS:
            continue

        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        x2 = min(x2, W)
        y2 = min(y2, H)
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue

        detections.append({
            "mode":       "detect",
            "bbox":       [x1, y1, x2, y2],
            "conf_det":   round(conf_d, 3),
            "label":      label,
            "confidence": round(conf_d, 4),
        })

    return detections, bool(detections)


def draw_detection_results(pil_img: Image.Image, detections: list,
                            class_colors: dict = None) -> Image.Image:
    """Vẽ bbox và label lên PIL Image, trả về ảnh mới."""
    if not detections:
        return pil_img
    if class_colors is None:
        class_colors = CLASS_COLORS

    out = pil_img.copy()
    draw = ImageDraw.Draw(out)

    try:
        font_label = ImageFont.truetype("C:\\Windows\\Fonts\\arialbd.ttf", 15)
        font_conf  = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 12)
    except Exception:
        font_label = ImageFont.load_default()
        font_conf  = font_label

    for det in detections:
        if det.get("mode") != "detect":
            continue
        x1, y1, x2, y2 = det["bbox"]
        label  = det["label"]
        conf_d = det["conf_det"]
        conf_c = det["confidence"]
        color  = class_colors.get(label, "#00e5ff")
        rgb    = _hex_to_rgb(color)

        for lw in range(3):
            draw.rectangle([x1 - lw, y1 - lw, x2 + lw, y2 + lw],
                           outline=rgb, width=1)

        tag_txt = f"{label}  {conf_c*100:.1f}%"
        try:
            bbox_txt = draw.textbbox((0, 0), tag_txt, font=font_label)
            tw = bbox_txt[2] - bbox_txt[0]
            th = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            tw, th = draw.textsize(tag_txt, font=font_label)

        pad = 4
        tag_y = max(0, y1 - th - pad * 2)
        draw.rectangle([x1, tag_y, x1 + tw + pad * 2, tag_y + th + pad * 2],
                       fill=(*rgb, 230))
        draw.text((x1 + pad, tag_y + pad), tag_txt,
                  fill=(255, 255, 255), font=font_label)

        det_txt = f"det {conf_d*100:.0f}%"
        draw.text((x1 + 4, y2 - 16), det_txt,
                  fill=(*rgb, 200), font=font_conf)

    return out


# ══════════════════════════════════════════════════════════════
# 3. GIAO DIỆN TKINTER
# ══════════════════════════════════════════════════════════════

class WasteClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WasteSense — Phát hiện & Phân loại Rác thải")
        self.geometry("1440x900")
        self.minsize(1180, 700)
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)

        # ── Chế độ phân tích hiện tại (1–4) ──
        self._analysis_mode = tk.IntVar(value=1)

        # ── Model cache: {mode_id: (yolo_model, cls_model, device, class_names, img_size)} ──
        self._model_cache: dict = {}
        self._loading_modes: set = set()

        # ── Model hiện tại ──
        self.cls_model   = None
        self.device      = None
        self.class_names = None
        self.img_size    = 224
        self.yolo_model  = None
        self.sahi_model  = None

        # ── Camera state ──
        self._cap            = None
        self._live           = False
        self._live_thread    = None
        self._last_frame     = None      # PIL Image gốc (chưa annotate)
        self._display_frame  = None      # PIL Image đang hiển thị
        self._last_results   = []
        self._had_detections = False
        self._inferencing    = False

        self._build_ui()
        # Nạp model cho chế độ mặc định (mode 1)
        self._switch_mode(1)

    # ─────────────────────── UI HELPERS ─────────────────────
    def _sep(self, parent, pady=0):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=pady)

    def _card(self, parent, title=None, subtitle=None):
        """Trả về (outer, body) hoặc (outer, body, hdr) nếu có title."""
        outer = tk.Frame(parent, bg=BORDER, padx=1, pady=1)
        card = tk.Frame(outer, bg=BG_CARD)
        card.pack(fill="both", expand=True)
        if title:
            hdr = tk.Frame(card, bg=BG_ELEVATED, height=44)
            hdr.pack(fill="x")
            hdr.pack_propagate(False)
            tk.Label(hdr, text=title, font=FONT_HEAD,
                     fg=TEXT_PRIMARY, bg=BG_ELEVATED).pack(
                side="left", padx=16, pady=10)
            if subtitle:
                tk.Label(hdr, text=subtitle, font=FONT_CAPTION,
                         fg=TEXT_MUTED, bg=BG_ELEVATED).pack(
                    side="right", padx=16)
            body = tk.Frame(card, bg=BG_CARD)
            body.pack(fill="both", expand=True)
            # Trả về 3 giá trị khi có title để caller có thể chèn widget vào hdr
            return outer, body, hdr
        return outer, card, None

    def _action_btn(self, parent, text, icon, bg, active_bg, command, col):
        btn = tk.Button(
            parent, text=f"  {icon}  {text}",
            font=(FONT_FAMILY, 10, "bold"),
            fg="white", bg=bg, activeforeground="white",
            activebackground=active_bg, relief="flat", cursor="hand2",
            bd=0, pady=13, command=command)
        btn.grid(row=0, column=col, sticky="ew",
                 padx=(0 if col == 0 else 4, 0 if col == 2 else 4))
        btn.bind("<Enter>", lambda e, b=btn, h=active_bg: b.config(bg=h))
        btn.bind("<Leave>", lambda e, b=btn, n=bg: b.config(bg=n))
        return btn

    def _status_chip(self, parent, label):
        chip = tk.Frame(parent, bg=BG_ELEVATED, padx=10, pady=5)
        chip.pack(side="left", padx=(0, 8))
        tk.Label(chip, text=label, font=FONT_CAPTION,
                 fg=TEXT_MUTED, bg=BG_ELEVATED).pack(side="left")
        lbl = tk.Label(chip, text="● Đang tải", font=FONT_SMALL,
                       fg=WARNING, bg=BG_ELEVATED)
        lbl.pack(side="left", padx=(6, 0))
        return lbl

    def _info_row(self, parent, key, val="—"):
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=key, font=FONT_SMALL,
                 fg=TEXT_MUTED, bg=BG_CARD, width=14, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=val, font=(FONT_FAMILY, 9, "bold"),
                       fg=TEXT_SECONDARY, bg=BG_CARD, anchor="w")
        lbl.pack(side="left", fill="x", expand=True)
        return lbl

    # ─────────────────────── BUILD UI ───────────────────────
    def _build_ui(self):
        # ── Header ──
        header = tk.Frame(self, bg=BG_SURFACE, height=78)
        header.pack(fill="x")
        header.pack_propagate(False)

        brand = tk.Frame(header, bg=BG_SURFACE)
        brand.pack(side="left", padx=24, pady=14)

        logo = tk.Frame(brand, bg=PRIMARY, width=42, height=42)
        logo.pack(side="left")
        logo.pack_propagate(False)
        tk.Label(logo, text="♻", font=(FONT_FAMILY, 18),
                 fg="white", bg=PRIMARY).place(relx=0.5, rely=0.5, anchor="center")

        title_blk = tk.Frame(brand, bg=BG_SURFACE)
        title_blk.pack(side="left", padx=(14, 0))
        tk.Label(title_blk, text="WasteSense AI",
                 font=FONT_TITLE, fg=TEXT_PRIMARY, bg=BG_SURFACE).pack(anchor="w")
        tk.Label(title_blk,
                 text="Hệ thống phát hiện & phân loại rác thải — YOLO + EfficientNet",
                 font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_SURFACE).pack(anchor="w")

        status_area = tk.Frame(header, bg=BG_SURFACE)
        status_area.pack(side="right", padx=24, pady=18)
        self.lbl_yolo_status  = self._status_chip(status_area, "Detector")
        self.lbl_model_status = self._status_chip(status_area, "Classifier")

        self._sep(self)

        # ── Pipeline strip ──
        pipeline = tk.Frame(self, bg=BG_ELEVATED, height=36)
        pipeline.pack(fill="x")
        pipeline.pack_propagate(False)
        self.lbl_pipeline_strip = tk.Label(
            pipeline,
            text="  Pipeline:  ① YOLO phát hiện vùng rác  →  ② EfficientNet phân loại vật thể",
            font=FONT_SMALL, fg=ACCENT, bg=BG_ELEVATED, anchor="w")
        self.lbl_pipeline_strip.pack(fill="both", expand=True, padx=20)

        # ── Body ──
        body = tk.Frame(self, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=3, minsize=520)
        body.columnconfigure(1, weight=2, minsize=380)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)

        self._sep(self)
        status_bar = tk.Frame(self, bg=BG_SURFACE, height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.lbl_status = tk.Label(
            status_bar,
            text="Sẵn sàng — chọn ảnh hoặc bật camera để bắt đầu",
            font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_SURFACE, anchor="w")
        self.lbl_status.pack(side="left", padx=16)

        self.lbl_infer = tk.Label(
            status_bar, text="", font=FONT_SMALL,
            fg=ACCENT, bg=BG_SURFACE)
        self.lbl_infer.pack(side="left", padx=8)

        self.lbl_process_time = tk.Label(
            status_bar, text="", font=FONT_SMALL,
            fg=PRIMARY, bg=BG_SURFACE)
        self.lbl_process_time.pack(side="right", padx=(8, 0))

        self.lbl_fps = tk.Label(
            status_bar, text="", font=FONT_SMALL,
            fg=TEXT_MUTED, bg=BG_SURFACE)
        self.lbl_fps.pack(side="right", padx=16)

    def _build_left_panel(self, parent):
        outer, body, _ = self._card(
            parent, title="Khung hình phân tích", subtitle="Viewport")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        viewport_wrap = tk.Frame(body, bg=BG_INPUT, padx=2, pady=2)
        viewport_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.canvas = tk.Canvas(viewport_wrap, bg=BG_INPUT, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # ── Footer row 1: badge + time ──
        vp_footer = tk.Frame(body, bg=BG_CARD)
        vp_footer.pack(fill="x", padx=12, pady=(0, 4))

        self.lbl_live_badge = tk.Label(
            vp_footer, text="● OFFLINE", font=FONT_CAPTION,
            fg=TEXT_MUTED, bg=BG_ELEVATED, padx=8, pady=3)
        self.lbl_live_badge.pack(side="left")

        self.lbl_vp_time = tk.Label(
            vp_footer, text="", font=FONT_CAPTION,
            fg=PRIMARY, bg=BG_CARD)
        self.lbl_vp_time.pack(side="right", padx=4)

        tk.Label(vp_footer,
                 text="Upload ảnh hoặc chụp webcam để phân tích",
                 font=FONT_CAPTION, fg=TEXT_MUTED, bg=BG_CARD).pack(
            side="right", padx=(0, 12))

        # ── Footer row 2: thanh thống kê phát hiện nổi bật ──
        self._stats_bar = tk.Frame(body, bg=BG_ELEVATED,
                                   highlightbackground=BORDER, highlightthickness=1)
        self._stats_bar.pack(fill="x", padx=12, pady=(0, 10))

        # Label phần tựa nằm bên trái
        self._stats_title = tk.Label(
            self._stats_bar,
            text="  Kết quả phân tích",
            font=(FONT_FAMILY, 9, "bold"),
            fg=TEXT_MUTED, bg=BG_ELEVATED, pady=6)
        self._stats_title.pack(side="left", padx=(6, 0))

        # Frame chứa các badge màu
        self._stats_badges = tk.Frame(self._stats_bar, bg=BG_ELEVATED)
        self._stats_badges.pack(side="left", fill="x", expand=True, padx=8, pady=4)

        # Label tổng số bên phải
        self._stats_total = tk.Label(
            self._stats_bar, text="",
            font=(FONT_FAMILY, 11, "bold"),
            fg=PRIMARY, bg=BG_ELEVATED, padx=12)
        self._stats_total.pack(side="right")

        self._placeholder_visible = True
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("placeholder")
        w = self.canvas.winfo_width() or 720
        h = self.canvas.winfo_height() or 480
        cx, cy = w // 2, h // 2

        margin = 40
        self.canvas.create_rectangle(
            margin, margin, w - margin, h - margin,
            outline=BORDER_LIGHT, dash=(8, 6), width=1, tags="placeholder")

        self.canvas.create_text(
            cx, cy - 36, text="♻", font=(FONT_FAMILY, 44),
            fill=BORDER_LIGHT, tags="placeholder")
        self.canvas.create_text(
            cx, cy + 16, text="Chưa có ảnh đầu vào",
            font=(FONT_FAMILY, 13, "bold"), fill=TEXT_SECONDARY, tags="placeholder")
        self.canvas.create_text(
            cx, cy + 44,
            text="Upload ảnh  ·  Chụp webcam  ·  Live camera",
            font=FONT_BODY, fill=TEXT_MUTED, tags="placeholder")

    def _on_canvas_resize(self, event):
        if self._placeholder_visible:
            self._draw_placeholder()
        elif not self._live:
            frame = (self._display_frame if self._display_frame is not None
                     else self._last_frame)
            if frame is not None:
                self._show_pil_on_canvas(frame)

    def _build_right_panel(self, parent):
        sidebar = tk.Frame(parent, bg=BG_MAIN)
        sidebar.grid(row=0, column=1, sticky="nsew")
        # row 0: controls, row 1: mode, row 2: results (expand), row 3: legend, row 4: info
        for r, w in [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0)]:
            sidebar.rowconfigure(r, weight=w)
        sidebar.columnconfigure(0, weight=1)

        # ── Actions ──
        act_outer, act_body, _ = self._card(sidebar, title="Điều khiển")
        act_outer.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        btn_row = tk.Frame(act_body, bg=BG_CARD)
        btn_row.pack(fill="x", padx=14, pady=10)
        btn_row.columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_upload = self._action_btn(
            btn_row, "Upload", "📁", PRIMARY_DARK, PRIMARY,
            self._on_upload, 0)
        self.btn_capture = self._action_btn(
            btn_row, "Chụp ảnh", "📸", ACCENT2, ACCENT,
            self._on_capture, 1)
        self.btn_camera = self._action_btn(
            btn_row, "Live", "🎥", "#14532d", PRIMARY_DARK,
            self._on_toggle_camera, 2)

        # ── Nút Reload ──
        self.btn_reload = tk.Button(
            btn_row, text="  🔄  Reload",
            font=(FONT_FAMILY, 10, "bold"),
            fg="white", bg="#1e3a5f", activeforeground="white",
            activebackground="#2563eb", relief="flat", cursor="hand2",
            bd=0, pady=13, command=self._on_reload)
        self.btn_reload.grid(row=0, column=3, sticky="ew", padx=(4, 0))
        self.btn_reload.bind("<Enter>", lambda e: self.btn_reload.config(bg="#2563eb"))
        self.btn_reload.bind("<Leave>", lambda e: self.btn_reload.config(bg="#1e3a5f"))

        # ── Chế độ phân tích (4 modes) ──
        mode_outer, mode_body, _ = self._card(sidebar, title="Chế độ phân tích")
        mode_outer.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        mode_inner = tk.Frame(mode_body, bg=BG_CARD)
        mode_inner.pack(fill="x", padx=14, pady=10)

        self._mode_radio_btns = {}
        mode_colors = {1: PRIMARY_DARK, 2: ACCENT2, 3: "#7c3aed", 4: "#b45309"}
        mode_icons  = {1: "🔍", 2: "🔬", 3: "⚡", 4: "🚀"}

        for mode_id, cfg in ANALYSIS_MODES.items():
            color     = mode_colors[mode_id]
            icon      = mode_icons[mode_id]
            frame     = tk.Frame(mode_inner, bg=BG_CARD)
            frame.pack(fill="x", pady=2)

            rb = tk.Radiobutton(
                frame,
                text=f"  {icon}  {cfg['name']}",
                variable=self._analysis_mode,
                value=mode_id,
                font=(FONT_FAMILY, 9, "bold"),
                fg=TEXT_SECONDARY,
                bg=BG_ELEVATED,
                selectcolor=color,
                activebackground=BG_ELEVATED,
                activeforeground=TEXT_PRIMARY,
                indicatoron=False,
                relief="flat", bd=0,
                cursor="hand2", pady=8, padx=12,
                command=lambda m=mode_id: self._switch_mode(m))
            rb.pack(side="left", fill="x", expand=True)
            self._mode_radio_btns[mode_id] = rb

            tk.Label(frame, text=cfg["desc"],
                     font=FONT_CAPTION, fg=TEXT_MUTED, bg=BG_CARD).pack(
                side="left", padx=8)

        self._update_mode_buttons()

        # ── Status tải model ──
        self.lbl_mode_loading = tk.Label(
            mode_inner, text="", font=FONT_CAPTION,
            fg=WARNING, bg=BG_CARD)
        self.lbl_mode_loading.pack(anchor="w", pady=(4, 0))

        # ── Results ──
        res_outer, res_body, res_hdr = self._card(sidebar, title="Kết quả nhận diện")
        res_outer.grid(row=2, column=0, sticky="nsew", pady=(0, 10))

        # Chèn label số lượng vật thể vào header card (bên phải)
        self.lbl_topclass = tk.Label(
            res_hdr, text="", font=(FONT_FAMILY, 10, "bold"),
            fg=PRIMARY, bg=BG_ELEVATED)
        self.lbl_topclass.pack(side="right", padx=16)

        scroll_frame = tk.Frame(res_body, bg=BG_CARD)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self._result_canvas = tk.Canvas(
            scroll_frame, bg=BG_CARD, highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(
            scroll_frame, orient="vertical",
            command=self._result_canvas.yview,
            bg=BG_ELEVATED, troughcolor=BG_CARD,
            activebackground=PRIMARY)
        self._result_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._result_canvas.pack(side="left", fill="both", expand=True)

        self.result_inner = tk.Frame(self._result_canvas, bg=BG_CARD)
        self._result_win = self._result_canvas.create_window(
            (0, 0), window=self.result_inner, anchor="nw")
        self.result_inner.bind(
            "<Configure>",
            lambda e: self._result_canvas.configure(
                scrollregion=self._result_canvas.bbox("all")))
        self._result_canvas.bind(
            "<Configure>",
            lambda e: self._result_canvas.itemconfig(
                self._result_win, width=e.width))

        self._result_rows = []
        self._show_results_empty()

        # ── Chú thích lớp rác (chỉ 5 lớp, không có Background) ──
        leg_outer, leg_body, _ = self._card(
            sidebar, title="Chú thích 5 loại rác", subtitle="Classifier")
        leg_outer.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.legend_grid = tk.Frame(leg_body, bg=BG_CARD)
        self.legend_grid.pack(fill="both", expand=True, padx=14, pady=10)
        self._build_class_legend()

        # ── System info ──
        info_outer, info_body, _ = self._card(sidebar, title="Thông tin hệ thống")
        info_outer.grid(row=4, column=0, sticky="ew")

        info_inner = tk.Frame(info_body, bg=BG_CARD)
        info_inner.pack(fill="x", padx=14, pady=10)
        self.lbl_info_mode    = self._info_row(info_inner, "Chế độ",    "—")
        self.lbl_info_yolo    = self._info_row(info_inner, "Detector",  "—")
        self.lbl_info_cls     = self._info_row(info_inner, "Classifier","—")
        self.lbl_info_device  = self._info_row(info_inner, "Thiết bị",  "—")
        self.lbl_info_classes = self._info_row(info_inner, "Số lớp",    "—")
        self.lbl_info_img     = self._info_row(info_inner, "Kích thước","—")

    # ── Legend chỉ hiển thị 5 lớp rác thực sự ──
    def _build_class_legend(self):
        for w in self.legend_grid.winfo_children():
            w.destroy()

        display_classes = (
            [c for c in (self.class_names or CLASSIFIER_CLASSES)
             if c != BACKGROUND_CLASS]
            or WASTE_CLASSES
        )

        # Cấu hình cột để các ô kéo dài đều nhau
        self.legend_grid.columnconfigure(0, weight=1)
        self.legend_grid.columnconfigure(1, weight=1)

        for i, name in enumerate(display_classes):
            color = CLASS_COLORS.get(name, ACCENT)
            meta  = CLASS_META.get(name, {"icon": "●", "vi": name})
            col   = i % 2
            row   = i // 2

            cell = tk.Frame(self.legend_grid, bg=BG_ELEVATED,
                            padx=8, pady=6)
            cell.grid(row=row, column=col, sticky="nsew",
                      padx=4, pady=4)

            # Thanh màu bên trái
            bar = tk.Frame(cell, bg=color, width=6)
            bar.pack(side="left", fill="y", padx=(0, 8))
            bar.pack_propagate(False)

            info = tk.Frame(cell, bg=BG_ELEVATED)
            info.pack(side="left", fill="x", expand=True)

            tk.Label(info,
                     text=f"{meta['icon']}  {meta['vi']}",
                     font=(FONT_FAMILY, 10, "bold"),
                     fg=color, bg=BG_ELEVATED,
                     anchor="w").pack(fill="x")
            tk.Label(info,
                     text=name,
                     font=(FONT_FAMILY, 8),
                     fg=TEXT_MUTED, bg=BG_ELEVATED,
                     anchor="w").pack(fill="x")

    def _update_mode_buttons(self):
        """Cập nhật màu sắc radio button theo chế độ đang chọn."""
        mode_colors = {1: PRIMARY_DARK, 2: ACCENT2, 3: "#7c3aed", 4: "#b45309"}
        current = self._analysis_mode.get()
        for mode_id, rb in self._mode_radio_btns.items():
            if mode_id == current:
                rb.config(bg=mode_colors[mode_id], fg="white")
            else:
                rb.config(bg=BG_ELEVATED, fg=TEXT_MUTED)

    def _show_results_empty(self):
        for w in self._result_rows:
            w.destroy()
        self._result_rows.clear()
        self.lbl_topclass.config(text="")
        self._update_stats_bar([], False)   # xóa thanh thống kê

        empty = tk.Frame(self.result_inner, bg=BG_ELEVATED, padx=16, pady=20)
        empty.pack(fill="x", pady=4)
        tk.Label(empty, text="Chưa có kết quả",
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=TEXT_MUTED, bg=BG_ELEVATED).pack()
        tk.Label(empty, text="Tải ảnh lên để bắt đầu phân tích",
                 font=FONT_CAPTION, fg=TEXT_MUTED, bg=BG_ELEVATED).pack(pady=(4, 0))
        self._result_rows.append(empty)

    # ── Thanh thống kê phát hiện bên dưới ảnh ──
    def _update_stats_bar(self, results, had_detections):
        """Cập nhật thanh thống kê số lượng vật thể phát hiện bên dưới viewport."""
        # Xóa badges cũ
        for w in self._stats_badges.winfo_children():
            w.destroy()

        if not results or not had_detections:
            if not results:
                self._stats_title.config(text="  Kết quả phân tích", fg=TEXT_MUTED)
                self._stats_total.config(text="")
                # Badge "chưa có kết quả"
                tk.Label(self._stats_badges,
                         text="Chưa phân tích",
                         font=FONT_CAPTION, fg=TEXT_MUTED,
                         bg=BG_ELEVATED).pack(side="left", padx=4)
            else:
                # Classify mode: hiển thị top label
                det_disp = [r for r in results if r.get("label") != BACKGROUND_CLASS]
                if det_disp:
                    top = det_disp[0]
                    color = CLASS_COLORS.get(top["label"], ACCENT)
                    self._stats_title.config(
                        text="  Phân loại", fg=ACCENT)
                    badge = tk.Frame(self._stats_badges, bg=color,
                                     padx=10, pady=3)
                    badge.pack(side="left", padx=4)
                    tk.Label(badge,
                             text=f"{top['label']}  {top['confidence']*100:.1f}%",
                             font=(FONT_FAMILY, 9, "bold"),
                             fg="white", bg=color).pack()
                    self._stats_total.config(text="")
            return

        # Detect mode: đếm số lượng theo từng loại
        label_counts = {}
        for r in results:
            if r.get("mode") == "detect":
                lbl = r["label"]
                label_counts[lbl] = label_counts.get(lbl, 0) + 1

        total = sum(label_counts.values())
        self._stats_title.config(
            text="  Phát hiện", fg=PRIMARY)
        self._stats_total.config(
            text=f"{total} vật thể",
            fg=PRIMARY)

        for label, count in sorted(label_counts.items(),
                                   key=lambda x: -x[1]):
            color = CLASS_COLORS.get(label, ACCENT)
            rgb   = _hex_to_rgb(color)
            # Badge: nền màu theo lớp
            badge = tk.Frame(self._stats_badges, bg=color,
                             padx=8, pady=3)
            badge.pack(side="left", padx=(0, 6), pady=2)
            tk.Label(badge,
                     text=f"{label}  {count}",
                     font=(FONT_FAMILY, 9, "bold"),
                     fg="white", bg=color).pack()

    def _fmt_elapsed(self, sec: float) -> str:
        if sec < 1.0:
            return f"{sec * 1000:.0f} ms"
        return f"{sec:.2f} s"

    def _set_process_time(self, elapsed):
        if elapsed is None:
            self.lbl_process_time.config(text="")
            self.lbl_vp_time.config(text="")
            return
        txt = f"⏱ Xử lý: {self._fmt_elapsed(elapsed)}"
        self.lbl_process_time.config(text=txt)
        self.lbl_vp_time.config(text=txt)

    def _set_inferencing(self, active: bool):
        self._inferencing = active
        self.lbl_infer.config(
            text="⏳ Đang phân tích..." if active else "")

    def _set_live_badge(self, live: bool):
        if live:
            self.lbl_live_badge.config(
                text="● LIVE", fg=PRIMARY_LIGHT, bg="#14532d")
        else:
            self.lbl_live_badge.config(
                text="● OFFLINE", fg=TEXT_MUTED, bg=BG_ELEVATED)

    # ─────────────────────── SWITCH MODE ─────────────────────
    def _switch_mode(self, mode_id: int):
        """Chuyển chế độ phân tích, nạp model nếu chưa có trong cache."""
        self._analysis_mode.set(mode_id)
        self._update_mode_buttons()

        cfg = ANALYSIS_MODES[mode_id]
        self.lbl_info_mode.config(text=cfg["name"])

        # Cập nhật pipeline strip
        if cfg["one_stage"]:
            self.lbl_pipeline_strip.config(
                text=f"  Pipeline [{cfg['name']}]:  YOLO phát hiện & phân loại 1 bước — {cfg['desc']}")
        else:
            self.lbl_pipeline_strip.config(
                text=f"  Pipeline [{cfg['name']}]:  ① YOLO phát hiện vùng rác  →  ② EfficientNet phân loại vật thể")

        if mode_id in self._model_cache:
            self._apply_cached_model(mode_id)
            return

        if mode_id in self._loading_modes:
            self._set_status(f"Đang tải model {cfg['name']}...")
            return

        self._loading_modes.add(mode_id)
        self._set_status(f"Đang tải model {cfg['name']}...")
        self.lbl_mode_loading.config(text=f"⏳ Đang tải {cfg['name']}...")
        self.lbl_yolo_status.config(text="● Đang tải", fg=WARNING)
        if cfg["pth"]:
            self.lbl_model_status.config(text="● Đang tải", fg=WARNING)
        else:
            self.lbl_model_status.config(text="● N/A (1-Stage)", fg=TEXT_MUTED)

        threading.Thread(
            target=self._load_mode_thread,
            args=(mode_id,), daemon=True).start()

    def _load_mode_thread(self, mode_id: int):
        cfg = ANALYSIS_MODES[mode_id]
        try:
            yolo = load_yolo_model(cfg["pt"])
            sahi_model = None
            if cfg.get("use_sahi") and HAS_SAHI:
                sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",
                    model_path=cfg["pt"],
                    confidence_threshold=0.3,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
        except Exception as e:
            self.after(0, lambda err=str(e): self._on_mode_load_error(mode_id, err))
            return

        cls_model  = None
        device     = None
        class_names = None
        img_size   = 224

        if cfg["pth"] and os.path.exists(cfg["pth"]):
            try:
                cls_model, device, class_names, img_size = load_cls_model(cfg["pth"])
            except Exception as e:
                self.after(0, lambda err=str(e): self._on_mode_load_error(mode_id, err))
                return
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model_cache[mode_id] = (yolo, cls_model, device, class_names, img_size, sahi_model)
        self._loading_modes.discard(mode_id)
        self.after(0, lambda m=mode_id: self._on_mode_loaded(m))

    def _on_mode_loaded(self, mode_id: int):
        # Chỉ áp dụng nếu đây vẫn là chế độ đang chọn
        if self._analysis_mode.get() == mode_id:
            self._apply_cached_model(mode_id)
        self.lbl_mode_loading.config(text="")

    def _on_mode_load_error(self, mode_id: int, err: str):
        self._loading_modes.discard(mode_id)
        self.lbl_mode_loading.config(text=f"❌ Lỗi tải mode {mode_id}")
        self.lbl_yolo_status.config(text="● Lỗi", fg=DANGER)
        self._set_status(f"Lỗi tải model: {err}")

    def _apply_cached_model(self, mode_id: int):
        yolo, cls_model, device, class_names, img_size, sahi_model = self._model_cache[mode_id]
        cfg = ANALYSIS_MODES[mode_id]

        self.yolo_model  = yolo
        self.cls_model   = cls_model
        self.device      = device
        self.class_names = class_names
        self.img_size    = img_size
        self.sahi_model  = sahi_model

        # Cập nhật status chips
        nc = len(yolo.names) if yolo else "?"
        self.lbl_yolo_status.config(text="● Sẵn sàng", fg=PRIMARY)
        if cls_model:
            self.lbl_model_status.config(text="● Sẵn sàng", fg=PRIMARY)
        else:
            self.lbl_model_status.config(text="● N/A (1-Stage)", fg=TEXT_MUTED)

        # Cập nhật info
        self.lbl_info_mode.config(text=cfg["name"])
        self.lbl_info_yolo.config(text=f"YOLO ({nc} lớp) — {os.path.basename(cfg['pt'])}")
        if cls_model:
            self.lbl_info_cls.config(text=f"EfficientNet — {os.path.basename(cfg['pth'])}")
        else:
            self.lbl_info_cls.config(text="— (1-Stage YOLO)")
        self.lbl_info_device.config(text=str(device).upper())
        self.lbl_info_classes.config(
            text=str(len(class_names)) if class_names else str(nc))
        self.lbl_info_img.config(text=f"{img_size}×{img_size}px")

        self._build_class_legend()
        self._set_status(f"Đã nạp {cfg['name']} — sẵn sàng phân tích")

    # ─────────────────────── MODEL LOADING (legacy stubs) ────
    def _on_cls_error(self, msg):
        self.lbl_model_status.config(text="● Lỗi", fg=DANGER)
        messagebox.showerror("Lỗi tải Classifier", msg)

    def _on_yolo_error(self, msg):
        self.lbl_yolo_status.config(text="● Lỗi", fg=DANGER)

    # ─────────────────────── 3 ACTIONS ──────────────────────

    # ACTION 1: Upload ảnh
    def _on_upload(self):
        self._stop_camera()
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("Tất cả", "*.*")])
        if not path:
            return
        self._current_image_path = path
        self._set_status(f"Đang xử lý: {os.path.basename(path)} ...")
        self.after(10, lambda: self._process_image_path(path))

    def _process_image_path(self, path: str):
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh:\n{e}")
            return
        self._last_frame = pil_img
        self._placeholder_visible = False
        self._show_pil_on_canvas(pil_img)
        self._run_inference(pil_img)
        self._set_status(f"Upload: {os.path.basename(path)}")

    # ACTION 2: Chụp ảnh
    def _on_capture(self):
        if self._live:
            if self._last_frame is not None:
                frame = self._last_frame.copy()
                self._stop_camera()
                self._run_inference(frame)
                self._set_status("Đã chụp ảnh từ camera")
        else:
            self._set_status("Đang mở camera để chụp ảnh...")
            threading.Thread(target=self._capture_single_frame,
                             daemon=True).start()

    def _capture_single_frame(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror(
                "Lỗi", "Không tìm thấy webcam."))
            self.after(0, lambda: self._set_status("Lỗi: Không có webcam"))
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.after(0, lambda: messagebox.showerror(
                "Lỗi", "Không thể đọc frame từ webcam."))
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(frame_rgb)
        self._last_frame = pil_img
        self._placeholder_visible = False
        self.after(0, lambda: self._show_pil_on_canvas(pil_img))
        self.after(0, lambda: self._run_inference(pil_img))
        self.after(0, lambda: self._set_status("Đã chụp ảnh từ webcam"))

    # ACTION 3: Live camera
    def _on_toggle_camera(self):
        if self._live:
            self._stop_camera()
        else:
            self._start_camera()

    # ACTION 4: Reload (chạy lại inference trên ảnh hiện tại)
    def _on_reload(self):
        if self._live:
            self._set_status("Reload không khả dụng khi đang chạy Live camera")
            return
        if self._last_frame is None:
            self._set_status("Chưa có ảnh để reload — hãy upload ảnh trước")
            return
        mode_id = self._analysis_mode.get()
        cfg = ANALYSIS_MODES[mode_id]

        # Nếu model chưa sẵn sàng → báo và chờ
        if mode_id not in self._model_cache:
            if mode_id in self._loading_modes:
                self._set_status(f"Đang tải model {cfg['name']}, vui lòng đợi...")
            else:
                self._switch_mode(mode_id)
                self._set_status(f"Đang tải model {cfg['name']}...")
            return

        self._apply_cached_model(mode_id)
        self._show_pil_on_canvas(self._last_frame)  # reset về ảnh gốc
        self._run_inference(self._last_frame)
        self._set_status(f"🔄 Reload với {cfg['name']}...")

    def _start_camera(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror(
                "Lỗi", "Không tìm thấy webcam.\nHãy kết nối webcam và thử lại.")
            return
        self._cap  = cap
        self._live = True
        self._placeholder_visible = False
        self._set_live_badge(True)
        self._set_process_time(None)
        self.btn_camera.config(
            text="  ⏹  Dừng Live",
            bg=DANGER, activebackground="#b91c1c")
        self._set_status("Live camera đang chạy — nhận diện real-time")
        self._live_thread = threading.Thread(
            target=self._live_loop, daemon=True)
        self._live_thread.start()

    def _stop_camera(self):
        if not self._live:
            return
        self._live = False
        time.sleep(0.12)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._set_live_badge(False)
        self.btn_camera.config(
            text="  🎥  Live",
            bg="#14532d", activebackground=PRIMARY_DARK)
        self.lbl_fps.config(text="")
        self._set_status("Camera đã dừng")

    def _live_loop(self):
        infer_every  = 8
        frame_count  = 0
        t_prev       = time.time()
        results_cache= []
        had_det_cache= False
        mode_id      = self._analysis_mode.get()

        while self._live and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            self._last_frame = pil_img
            frame_count += 1

            cur_mode = self._analysis_mode.get()
            cfg = ANALYSIS_MODES[cur_mode]

            if frame_count % infer_every == 0:
                try:
                    if cfg.get("use_sahi") and HAS_SAHI and self.sahi_model:
                        if cfg["one_stage"]:
                            res, had = predict_one_stage_sahi(self.sahi_model, pil_img)
                        else:
                            res, had = predict_with_detection_sahi(
                                self.sahi_model, self.cls_model, self.device,
                                pil_img, self.img_size, self.class_names)
                    else:
                        if cfg["one_stage"] and self.yolo_model:
                            res, had = predict_one_stage(self.yolo_model, pil_img)
                        elif (not cfg["one_stage"]
                              and self.yolo_model and self.cls_model):
                            res, had = predict_with_detection(
                                self.yolo_model, self.cls_model, self.device,
                                pil_img, self.img_size, self.class_names)
                        else:
                            res, had = [], False
                    results_cache  = res
                    had_det_cache  = had
                    self._last_results   = res
                    self._had_detections = had
                except Exception:
                    pass

            now  = time.time()
            fps  = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            display_img = self._draw_live_overlay(
                pil_img.copy(), results_cache, had_det_cache, fps)

            self.after(0, lambda img=display_img, r=list(results_cache),
                       hd=had_det_cache, f=fps:
                       self._update_live_ui(img, r, hd, f))
            time.sleep(0.033)

        self.after(0, lambda: self._set_status("Camera đã dừng"))

    def _draw_live_overlay(self, pil_img, results, had_detections, fps):
        if not results:
            return pil_img

        if had_detections:
            det_only = [r for r in results if r.get("mode") == "detect"]
            pil_img  = draw_detection_results(pil_img, det_only, CLASS_COLORS)

        draw = ImageDraw.Draw(pil_img, "RGBA")
        try:
            font_s = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 13)
            font_b = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 15)
        except Exception:
            font_s = font_b = ImageFont.load_default()

        if had_detections:
            label_counts = {}
            for r in results:
                if r.get("mode") == "detect":
                    lbl = r["label"]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            lines = [f"{lbl}: {cnt}" for lbl, cnt in label_counts.items()]
            n_obj = sum(label_counts.values())
            title = f"DETECT ({n_obj} vật thể)"
        else:
            lines = [f"{r['label']}  {r['confidence']*100:.1f}%"
                     for r in results[:4]]
            title = "CLASSIFY"

        box_h = 30 + len(lines) * 22
        draw.rectangle([8, 8, 270, 8 + box_h], fill=(12, 18, 16, 210))
        draw.rectangle([8, 8, 270, 8 + box_h],
                       outline=(34, 197, 94, 255), width=1)
        draw.text((16, 13), title, fill=(45, 212, 191, 255), font=font_b)
        y = 33
        for line in lines:
            draw.text((16, y), line, fill=(236, 253, 245, 230), font=font_s)
            y += 22

        draw.text((pil_img.width - 85, 14),
                  f"FPS {fps:.1f}",
                  fill=(74, 222, 128, 200), font=font_s)
        return pil_img

    def _update_live_ui(self, pil_img, results, had_detections, fps=0):
        self._show_pil_on_canvas(pil_img)
        self.lbl_fps.config(text=f"FPS {fps:.1f}" if fps else "")
        if results:
            self._render_results(results, had_detections)

    # ─────────────────────── INFERENCE ──────────────────────
    def _run_inference(self, pil_img: Image.Image):
        mode_id = self._analysis_mode.get()
        cfg = ANALYSIS_MODES[mode_id]

        if cfg["one_stage"]:
            if self.yolo_model is None:
                messagebox.showwarning(
                    "Chưa tải model",
                    f"Model {cfg['name']} chưa sẵn sàng. Vui lòng đợi trạng thái ● Sẵn sàng.")
                return
        else:
            if self.cls_model is None or self.yolo_model is None:
                messagebox.showwarning(
                    "Chưa tải model",
                    f"Model {cfg['name']} chưa sẵn sàng. Vui lòng đợi trạng thái ● Sẵn sàng.")
                return

        self._set_inferencing(True)
        self._set_process_time(None)
        self._set_status("Đang phân tích ảnh...")
        threading.Thread(target=self._infer_thread,
                         args=(pil_img, mode_id), daemon=True).start()

    def _infer_thread(self, pil_img, mode_id):
        t_start = time.perf_counter()
        cfg = ANALYSIS_MODES[mode_id]
        try:
            if cfg.get("use_sahi") and HAS_SAHI and self.sahi_model:
                if cfg["one_stage"]:
                    results, had_det = predict_one_stage_sahi(self.sahi_model, pil_img)
                else:
                    results, had_det = predict_with_detection_sahi(
                        self.sahi_model, self.cls_model, self.device,
                        pil_img, self.img_size, self.class_names)
            else:
                if cfg["one_stage"]:
                    results, had_det = predict_one_stage(self.yolo_model, pil_img)
                else:
                    results, had_det = predict_with_detection(
                        self.yolo_model, self.cls_model, self.device,
                        pil_img, self.img_size, self.class_names)

            self._last_results   = results
            self._had_detections = had_det

            elapsed = time.perf_counter() - t_start

            if had_det:
                det_only = [r for r in results if r.get("mode") == "detect"]
                annotated = draw_detection_results(pil_img.copy(),
                                                   det_only, CLASS_COLORS)
                elapsed = time.perf_counter() - t_start
                n_obj = len(det_only)
                self.after(0, lambda img=annotated: self._show_pil_on_canvas(img))
                self.after(0, lambda r=results, hd=had_det:
                           self._render_results(r, hd))
                self.after(0, lambda n=n_obj:
                           self._set_status(f"Phát hiện {n} vật thể"))
            else:
                self.after(0, lambda r=results, hd=had_det:
                           self._render_results(r, hd))
                msg = ("Không phát hiện rác → phân loại cả ảnh"
                       if not cfg["one_stage"] else "Không phát hiện vật thể nào")
                self.after(0, lambda m=msg: self._set_status(m))

            self.after(0, lambda e=elapsed: self._set_process_time(e))

        except Exception as e:
            self.after(0, lambda err=e: self._set_status(f"Lỗi: {err}"))
        finally:
            self.after(0, lambda: self._set_inferencing(False))

    # ─────────────────────── RENDER RESULTS ─────────────────
    def _render_results(self, results, had_detections=False):
        for w in self._result_rows:
            w.destroy()
        self._result_rows.clear()

        # Luôn cập nhật thanh thống kê bên dưới ảnh
        self._update_stats_bar(results, had_detections)

        if not results:
            self._show_results_empty()
            return

        if had_detections:
            det = [r for r in results if r.get("mode") == "detect"]
            n_obj = len(det)
            self.lbl_topclass.config(
                text=f"{n_obj} vật thể", fg=PRIMARY)
            for i, r in enumerate(det[:8], 1):
                row = self._make_detect_row(self.result_inner, i, r)
                row.pack(fill="x", pady=4)
                self._result_rows.append(row)
        else:
            results_disp = [r for r in results if r["label"] != BACKGROUND_CLASS]
            if not results_disp:
                results_disp = results

            top = results_disp[0]
            color = CLASS_COLORS.get(top["label"], ACCENT)
            meta  = CLASS_META.get(top["label"], {})
            vi    = meta.get("vi", top["label"])
            self.lbl_topclass.config(
                text=f"{vi} · {top['confidence']*100:.1f}%", fg=color)
            for rank, r in enumerate(results_disp, 1):
                row = self._make_cls_row(self.result_inner, rank, r)
                row.pack(fill="x", pady=4)
                self._result_rows.append(row)

        self._result_canvas.configure(
            scrollregion=self._result_canvas.bbox("all"))

    def _make_detect_row(self, parent, rank, r):
        label  = r["label"]
        conf_d = r["conf_det"]
        conf_c = r["confidence"]
        bbox   = r["bbox"]
        color  = CLASS_COLORS.get(label, ACCENT)
        meta   = CLASS_META.get(label, {"vi": label})
        is_top = rank == 1

        row = tk.Frame(parent, bg=BG_ELEVATED if is_top else BG_CARD,
                       highlightbackground=color if is_top else BORDER,
                       highlightthickness=2 if is_top else 1)

        accent_bar = tk.Frame(row, bg=color, width=4)
        accent_bar.pack(side="left", fill="y")

        content = tk.Frame(row, bg=BG_ELEVATED if is_top else BG_CARD)
        content.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        hdr = tk.Frame(content, bg=BG_ELEVATED if is_top else BG_CARD)
        hdr.pack(fill="x")

        tk.Label(hdr, text=f"#{rank}",
                 font=(FONT_FAMILY, 8, "bold"),
                 fg=BG_MAIN if is_top else TEXT_MUTED,
                 bg=color if is_top else BG_CARD,
                 padx=6, pady=1).pack(side="left")

        tk.Label(hdr, text=f"  {meta['vi']}",
                 font=(FONT_FAMILY, 11, "bold" if is_top else "normal"),
                 fg=color if is_top else TEXT_PRIMARY,
                 bg=BG_ELEVATED if is_top else BG_CARD).pack(side="left")

        tk.Label(hdr, text=f"{conf_c*100:.1f}%",
                 font=(FONT_FAMILY, 11, "bold"),
                 fg=color, bg=BG_ELEVATED if is_top else BG_CARD).pack(side="right")

        sub = tk.Frame(content, bg=BG_ELEVATED if is_top else BG_CARD)
        sub.pack(fill="x", pady=(4, 0))
        tk.Label(sub,
                 text=f"YOLO {conf_d*100:.0f}%  ·  [{bbox[0]}, {bbox[1]}] → [{bbox[2]}, {bbox[3]}]",
                 font=FONT_CAPTION, fg=TEXT_MUTED,
                 bg=BG_ELEVATED if is_top else BG_CARD).pack(anchor="w")

        return row

    def _make_cls_row(self, parent, rank, r):
        conf  = r["confidence"]
        label = r["label"]
        color = CLASS_COLORS.get(label, ACCENT)
        meta  = CLASS_META.get(label, {"vi": label})
        is_top = rank == 1

        row = tk.Frame(parent, bg=BG_ELEVATED if is_top else BG_CARD,
                       highlightbackground=color if is_top else BORDER,
                       highlightthickness=2 if is_top else 1)

        accent_bar = tk.Frame(row, bg=color, width=4)
        accent_bar.pack(side="left", fill="y")

        content = tk.Frame(row, bg=BG_ELEVATED if is_top else BG_CARD)
        content.pack(side="left", fill="both", expand=True, padx=10, pady=8)

        hdr = tk.Frame(content, bg=BG_ELEVATED if is_top else BG_CARD)
        hdr.pack(fill="x")

        tk.Label(hdr, text=f"#{rank}",
                 font=(FONT_FAMILY, 8, "bold"),
                 fg=BG_MAIN if is_top else TEXT_MUTED,
                 bg=color if is_top else BG_CARD,
                 padx=6, pady=1).pack(side="left")

        tk.Label(hdr, text=f"  {meta['vi']}",
                 font=(FONT_FAMILY, 11, "bold" if is_top else "normal"),
                 fg=color if is_top else TEXT_PRIMARY,
                 bg=BG_ELEVATED if is_top else BG_CARD).pack(side="left")

        tk.Label(hdr, text=f"{conf*100:.1f}%",
                 font=(FONT_FAMILY, 11, "bold"),
                 fg=color, bg=BG_ELEVATED if is_top else BG_CARD).pack(side="right")

        bar_bg = tk.Frame(content, bg=BORDER, height=5)
        bar_bg.pack(fill="x", pady=(6, 0))
        bar_bg.pack_propagate(False)
        tk.Frame(bar_bg, bg=color, height=5).place(relwidth=conf, relheight=1.0)

        return row

    # ─────────────────────── DISPLAY HELPERS ────────────────
    def _show_pil_on_canvas(self, pil_img: Image.Image):
        self._display_frame = pil_img
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img_w, img_h = pil_img.size
        scale = min(cw / img_w, ch / img_h)
        nw = max(1, int(img_w * scale))
        nh = max(1, int(img_h * scale))
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2,
                                 anchor="center", image=self._tk_img)

    def _set_status(self, msg: str):
        self.lbl_status.config(text=msg)

    # ─────────────────────── CLEANUP ────────────────────────
    def on_close(self):
        self._stop_camera()
        self.destroy()


# ══════════════════════════════════════════════════════════════
# 4. ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    app = WasteClassifierApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()