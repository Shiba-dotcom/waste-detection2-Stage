# -*- coding: utf-8 -*-
"""
Demo GUI - Waste Detection & Classification
Pipeline 2 bước:
  1. YOLO best.pt → phát hiện bbox vùng rác
  2. EfficientNet stage2_best.pth → phân loại từng crop

Giao diện với 3 chế độ:
  1. Upload hình ảnh
  2. Chụp ảnh từ webcam
  3. Nhận diện video trực tiếp (Live Camera)
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

# ══════════════════════════════════════════════════════════════
# 0. CẤU HÌNH MẶC ĐỊNH
# ══════════════════════════════════════════════════════════════

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CLS_MODEL_PATH    = os.path.join(_BASE, "models", "sahi_stage2.pth")
DEFAULT_DETECT_MODEL_PATH = os.path.join(_BASE, "models", "sahi_best.pt")
OUTPUT_DIR = os.path.join(_BASE, "results")

# ── Theme: Eco / Waste Classification (đề tài phân loại rác thải) ──
BG_MAIN       = "#0c1210"      # nền chính — đen xanh lá
BG_SURFACE    = "#141f1a"      # nền vùng nội dung
BG_CARD       = "#1a2822"      # card
BG_ELEVATED   = "#223329"      # card nổi / header con
BG_INPUT      = "#0a100e"      # viewport ảnh

PRIMARY       = "#22c55e"      # xanh lá — tái chế / môi trường
PRIMARY_DARK  = "#16a34a"
PRIMARY_LIGHT = "#4ade80"
ACCENT        = "#2dd4bf"      # teal — công nghệ / AI
ACCENT2       = "#0d9488"
WARNING       = "#f59e0b"
DANGER        = "#ef4444"
TEXT_PRIMARY  = "#ecfdf5"
TEXT_SECONDARY= "#cbd5e1"
TEXT_MUTED    = "#64748b"
BORDER        = "#2a3f35"
BORDER_LIGHT  = "#3d5248"

# Alias tương thích code cũ
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

# 6 lớp EfficientNet classifier (đúng thứ tự checkpoint stage2_best.pth)
CLASSIFIER_CLASSES = [
    "Background", "Glass", "Metal", "Other", "Paper", "Plastic",
]

CLASS_META = {
    "Background": {"icon": "◻", "vi": "Nền / không phải rác"},
    "Glass":      {"icon": "◆", "vi": "Thủy tinh"},
    "Metal":      {"icon": "▣", "vi": "Kim loại"},
    "Other":      {"icon": "●", "vi": "Rác khác"},
    "Paper":      {"icon": "▤", "vi": "Giấy"},
    "Plastic":    {"icon": "▲", "vi": "Nhựa"},
    "Waste":      {"icon": "♻", "vi": "Vùng rác (YOLO)"},  # chỉ dùng khi vẽ bbox
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

# Chuyển hex → RGB tuple
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
    
    # Init SAHI wrapper
    try:
        from sahi import AutoDetectionModel
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=pt_path,
            confidence_threshold=0.25,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        model.sahi_model = sahi_model
    except Exception as e:
        print("Không thể load SAHI:", e)
        model.sahi_model = None

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
                            class_names, conf_thresh: float = 0.25):
    """
    Pipeline 2 bước:
      1. YOLO → detect bbox vùng rác
      2. EfficientNet → classify từng crop bbox → lấy top-1
    """
    import numpy as np
    img_np = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    W, H = pil_img.size
    detections = []
    
    sahi_model = getattr(yolo_model, 'sahi_model', None)
    
    if sahi_model is not None:
        from sahi.predict import get_sliced_prediction
        sahi_model.confidence_threshold = conf_thresh
        result = get_sliced_prediction(
            img_np,
            sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        for obj in result.object_prediction_list:
            bbox = obj.bbox.to_xyxy()
            conf_d = float(obj.score.value)
            
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

            detections.append({
                "mode":       "detect",
                "bbox":       [x1, y1, x2, y2],
                "conf_det":   round(conf_d, 3),
                "label":      top_label,
                "confidence": round(top_conf, 4),
            })
    else:
        # Standard YOLO inference
        results_yolo = yolo_model(img_np, conf=conf_thresh, verbose=False)
        for r in results_yolo:
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                xyxy  = box.xyxy[0].cpu().tolist()   # [x1, y1, x2, y2]
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
    
                detections.append({
                    "mode":       "detect",
                    "bbox":       [x1, y1, x2, y2],
                    "conf_det":   round(conf_d, 3),
                    "label":      top_label,
                    "confidence": round(top_conf, 4),
                })

    # Fallback nếu YOLO không detect được gì
    if not detections:
        cls_res = predict_cls(cls_model, device, pil_img,
                              img_size, class_names, top_k=5)
        return cls_res, False   # (results, had_detections)

    return detections, True     # (results, had_detections)


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
        label   = det["label"]
        conf_d  = det["conf_det"]
        conf_c  = det["confidence"]
        color   = class_colors.get(label, "#00e5ff")
        rgb     = _hex_to_rgb(color)

        # Vẽ bbox
        for lw in range(3):
            draw.rectangle([x1 - lw, y1 - lw, x2 + lw, y2 + lw],
                           outline=rgb, width=1)

        # Nền nhãn
        tag_txt = f"{label}  {conf_c*100:.1f}%"
        try:
            bbox_txt = draw.textbbox((0, 0), tag_txt, font=font_label)
            tw = bbox_txt[2] - bbox_txt[0]
            th = bbox_txt[3] - bbox_txt[1]
        except AttributeError:
            tw, th = draw.textsize(tag_txt, font=font_label)  # PIL < 10

        pad = 4
        tag_y = max(0, y1 - th - pad * 2)
        draw.rectangle([x1, tag_y, x1 + tw + pad * 2, tag_y + th + pad * 2],
                       fill=(*rgb, 230))
        draw.text((x1 + pad, tag_y + pad), tag_txt,
                  fill=(255, 255, 255), font=font_label)

        # Nhỏ: confidence YOLO ở góc dưới-phải bbox
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
        self.geometry("1440x860")
        self.minsize(1180, 680)
        self.configure(bg=BG_MAIN)
        self.resizable(True, True)

        # ── State: Classification model ──
        self.cls_model   = None
        self.device      = None
        self.class_names = None
        self.img_size    = 224

        # ── State: YOLO detection model ──
        self.yolo_model    = None
        self.use_detection = True

        # ── Camera state ──
        self._cap           = None
        self._live          = False
        self._live_thread   = None
        self._last_frame    = None
        self._display_frame = None
        self._last_results  = []
        self._had_detections = False
        self._inferencing   = False

        self._build_ui()
        self._load_all_models()

    # ─────────────────────── UI HELPERS ─────────────────────
    def _sep(self, parent, pady=0):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=pady)

    def _card(self, parent, title=None, subtitle=None):
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
            return outer, body
        return outer, card

    def _action_btn(self, parent, text, icon, bg, active_bg, command, col):
        btn = tk.Button(
            parent, text=f"  {icon}  {text}",
            font=(FONT_FAMILY, 10, "bold"),
            fg="white", bg=bg, activeforeground="white",
            activebackground=active_bg, relief="flat", cursor="hand2",
            bd=0, pady=13, command=command)
        btn.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 4, 0 if col == 2 else 4))
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
        self.lbl_yolo_status = self._status_chip(status_area, "Detector")
        self.lbl_model_status = self._status_chip(status_area, "Classifier")

        self._sep(self)

        # ── Pipeline strip ──
        pipeline = tk.Frame(self, bg=BG_ELEVATED, height=36)
        pipeline.pack(fill="x")
        pipeline.pack_propagate(False)
        tk.Label(pipeline,
                 text="  Pipeline:  ① YOLO phát hiện vùng rác  →  ② EfficientNet phân loại vật thể",
                 font=FONT_SMALL, fg=ACCENT, bg=BG_ELEVATED, anchor="w").pack(
            fill="both", expand=True, padx=20)

        # ── Body ──
        body = tk.Frame(self, bg=BG_MAIN)
        body.pack(fill="both", expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=3, minsize=520)
        body.columnconfigure(1, weight=2, minsize=360)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_right_panel(body)

        self._sep(self)
        status_bar = tk.Frame(self, bg=BG_SURFACE, height=32)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.lbl_status = tk.Label(
            status_bar, text="Sẵn sàng — chọn ảnh hoặc bật camera để bắt đầu",
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
        outer, body = self._card(
            parent, title="Khung hình phân tích", subtitle="Viewport")
        outer.grid(row=0, column=0, sticky="nsew", padx=(0, 12))

        viewport_wrap = tk.Frame(body, bg=BG_INPUT, padx=2, pady=2)
        viewport_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.canvas = tk.Canvas(viewport_wrap, bg=BG_INPUT, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Footer viewport: live badge + hint
        vp_footer = tk.Frame(body, bg=BG_CARD)
        vp_footer.pack(fill="x", padx=12, pady=(0, 10))

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

        self._placeholder_visible = True
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.canvas.delete("placeholder")
        w = self.canvas.winfo_width() or 720
        h = self.canvas.winfo_height() or 480
        cx, cy = w // 2, h // 2

        # Khung dashed giả lập
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
        for r, w in [(0, 0), (1, 0), (2, 1), (3, 0), (4, 0)]:
            sidebar.rowconfigure(r, weight=w)
        sidebar.columnconfigure(0, weight=1)

        # ── Actions ──
        act_outer, act_body = self._card(sidebar, title="Điều khiển")
        act_outer.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        btn_row = tk.Frame(act_body, bg=BG_CARD)
        btn_row.pack(fill="x", padx=14, pady=12)
        btn_row.columnconfigure((0, 1, 2), weight=1)

        self.btn_upload = self._action_btn(
            btn_row, "Upload", "📁", PRIMARY_DARK, PRIMARY,
            self._on_upload, 0)
        self.btn_capture = self._action_btn(
            btn_row, "Chụp ảnh", "📸", ACCENT2, ACCENT,
            self._on_capture, 1)
        self.btn_camera = self._action_btn(
            btn_row, "Live", "🎥", "#14532d", PRIMARY_DARK,
            self._on_toggle_camera, 2)

        # ── Mode toggle ──
        mode_outer, mode_body = self._card(sidebar, title="Chế độ phân tích")
        mode_outer.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        mode_inner = tk.Frame(mode_body, bg=BG_CARD)
        mode_inner.pack(fill="x", padx=14, pady=12)

        self._det_var = tk.BooleanVar(value=True)
        modes = tk.Frame(mode_inner, bg=BG_ELEVATED, padx=4, pady=4)
        modes.pack(fill="x")

        def _pick_mode(detect: bool):
            self._det_var.set(detect)
            self._on_toggle_detection()
            btn_det.config(bg=PRIMARY if detect else BG_ELEVATED,
                           fg="white" if detect else TEXT_MUTED)
            btn_cls.config(bg=ACCENT2 if not detect else BG_ELEVATED,
                           fg="white" if not detect else TEXT_MUTED)

        btn_det = tk.Button(
            modes, text="  🔍  Phát hiện + Phân loại",
            font=(FONT_FAMILY, 9, "bold"), relief="flat", bd=0,
            cursor="hand2", pady=10, bg=PRIMARY, fg="white",
            command=lambda: _pick_mode(True))
        btn_det.pack(side="left", fill="x", expand=True, padx=(0, 2))

        btn_cls = tk.Button(
            modes, text="  🏷  Chỉ phân loại",
            font=(FONT_FAMILY, 9, "bold"), relief="flat", bd=0,
            cursor="hand2", pady=10, bg=BG_ELEVATED, fg=TEXT_MUTED,
            command=lambda: _pick_mode(False))
        btn_cls.pack(side="left", fill="x", expand=True, padx=(2, 0))

        self._mode_btn_det = btn_det
        self._mode_btn_cls = btn_cls

        tk.Label(mode_inner,
                 text="YOLO định vị bbox → EfficientNet gán nhãn từng vùng",
                 font=FONT_CAPTION, fg=TEXT_MUTED, bg=BG_CARD).pack(
            anchor="w", pady=(8, 0))

        # ── Results (scrollable) ──
        res_outer, res_body = self._card(sidebar, title="Kết quả nhận diện")
        res_outer.grid(row=2, column=0, sticky="nsew", pady=(0, 10))

        res_hdr = res_body.master.winfo_children()[0]
        self.lbl_topclass = tk.Label(
            res_hdr, text="", font=FONT_SMALL,
            fg=ACCENT, bg=BG_ELEVATED)
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

        # ── Class legend (6 lớp classifier) ──
        leg_outer, leg_body = self._card(
            sidebar, title="Chú thích 6 loại rác", subtitle="EfficientNet")
        leg_outer.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        self.legend_grid = tk.Frame(leg_body, bg=BG_CARD)
        self.legend_grid.pack(fill="x", padx=14, pady=10)
        self._build_class_legend()

        # ── System info ──
        info_outer, info_body = self._card(sidebar, title="Thông tin hệ thống")
        info_outer.grid(row=4, column=0, sticky="ew")

        info_inner = tk.Frame(info_body, bg=BG_CARD)
        info_inner.pack(fill="x", padx=14, pady=10)
        self.lbl_info_yolo    = self._info_row(info_inner, "Detector", "—")
        self.lbl_info_cls     = self._info_row(info_inner, "Classifier", "—")
        self.lbl_info_device  = self._info_row(info_inner, "Thiết bị", "—")
        self.lbl_info_classes = self._info_row(info_inner, "Số lớp", "—")
        self.lbl_info_img     = self._info_row(info_inner, "Kích thước", "—")

    def _build_class_legend(self):
        """Vẽ chú thích đúng 6 lớp classifier."""
        for w in self.legend_grid.winfo_children():
            w.destroy()

        classes = (self.class_names if self.class_names
                   else CLASSIFIER_CLASSES)
        cols = 2
        for i, name in enumerate(classes):
            color = CLASS_COLORS.get(name, ACCENT)
            meta = CLASS_META.get(name, {"icon": "●", "vi": name})
            cell = tk.Frame(self.legend_grid, bg=BG_CARD)
            cell.grid(row=i // cols, column=i % cols,
                      sticky="w", padx=4, pady=3)
            swatch = tk.Frame(cell, bg=color, width=10, height=10)
            swatch.pack(side="left", padx=(0, 6))
            swatch.pack_propagate(False)
            tk.Label(
                cell,
                text=f"{meta['icon']}  {meta['vi']}",
                font=FONT_CAPTION, fg=TEXT_SECONDARY, bg=BG_CARD,
            ).pack(side="left")

    def _show_results_empty(self):
        for w in self._result_rows:
            w.destroy()
        self._result_rows.clear()
        self.lbl_topclass.config(text="")

        empty = tk.Frame(self.result_inner, bg=BG_ELEVATED, padx=16, pady=20)
        empty.pack(fill="x", pady=4)
        tk.Label(empty, text="Chưa có kết quả",
                 font=(FONT_FAMILY, 10, "bold"),
                 fg=TEXT_MUTED, bg=BG_ELEVATED).pack()
        tk.Label(empty, text="Tải ảnh lên để bắt đầu phân tích",
                 font=FONT_CAPTION, fg=TEXT_MUTED, bg=BG_ELEVATED).pack(pady=(4, 0))
        self._result_rows.append(empty)

    def _on_toggle_detection(self):
        self.use_detection = self._det_var.get()
        mode_txt = ("Phát hiện + Phân loại" if self.use_detection
                    else "Chỉ phân loại")
        self._set_status(f"Chế độ: {mode_txt}")

    def _fmt_elapsed(self, sec: float) -> str:
        if sec < 1.0:
            return f"{sec * 1000:.0f} ms"
        return f"{sec:.2f} s"

    def _set_process_time(self, elapsed: float | None):
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

    # ─────────────────────── MODEL LOADING ──────────────────
    def _load_all_models(self):
        """Load cả 2 model trong background threads."""
        if os.path.exists(DEFAULT_CLS_MODEL_PATH):
            threading.Thread(
                target=self._load_cls_thread,
                args=(DEFAULT_CLS_MODEL_PATH,), daemon=True).start()
        else:
            self.lbl_model_status.config(text="● Thiếu file", fg=DANGER)

        if os.path.exists(DEFAULT_DETECT_MODEL_PATH):
            threading.Thread(
                target=self._load_yolo_thread,
                args=(DEFAULT_DETECT_MODEL_PATH,), daemon=True).start()
        else:
            self.lbl_yolo_status.config(text="● Thiếu file", fg=DANGER)

    def _load_cls_thread(self, path):
        try:
            model, device, class_names, img_size = load_cls_model(path)
            self.cls_model   = model
            self.device      = device
            self.class_names = class_names
            self.img_size    = img_size
            self.after(0, self._on_cls_loaded, path)
        except Exception as e:
            self.after(0, self._on_cls_error, str(e))

    def _load_yolo_thread(self, path):
        try:
            yolo = load_yolo_model(path)
            self.yolo_model = yolo
            self.after(0, self._on_yolo_loaded, path)
        except Exception as e:
            self.after(0, self._on_yolo_error, str(e))

    def _on_cls_loaded(self, path):
        self.lbl_model_status.config(text="● Sẵn sàng", fg=PRIMARY)
        self.lbl_info_cls.config(text="EfficientNet-B2")
        self.lbl_info_device.config(text=str(self.device).upper())
        self.lbl_info_classes.config(
            text=str(len(self.class_names)) if self.class_names else "?")
        self.lbl_info_img.config(text=f"{self.img_size}×{self.img_size}px")
        self._build_class_legend()
        self._set_status(f"Classifier đã tải — {os.path.basename(path)}")

    def _on_yolo_loaded(self, path):
        self.lbl_yolo_status.config(text="● Sẵn sàng", fg=PRIMARY)
        nc = len(self.yolo_model.names) if self.yolo_model else "?"
        self.lbl_info_yolo.config(text=f"YOLO ({nc} lớp)")
        self._set_status(f"Detector đã tải — {os.path.basename(path)}")

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
        pil_img = Image.fromarray(frame_rgb)
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
        """Thread riêng: đọc frame, nhận diện, cập nhật UI."""
        infer_every = 8
        frame_count = 0
        t_prev = time.time()
        results_cache = []
        had_det_cache = False

        while self._live and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            self._last_frame = pil_img
            frame_count += 1

            if frame_count % infer_every == 0:
                if self.use_detection and self.yolo_model and self.cls_model:
                    try:
                        res, had = predict_with_detection(
                            self.yolo_model, self.cls_model, self.device,
                            pil_img, self.img_size, self.class_names)
                        results_cache  = res
                        had_det_cache  = had
                        self._last_results     = res
                        self._had_detections   = had
                    except Exception:
                        pass
                elif self.cls_model:
                    try:
                        res = predict_cls(
                            self.cls_model, self.device, pil_img,
                            self.img_size, self.class_names, top_k=5)
                        results_cache = res
                        had_det_cache = False
                        self._last_results   = res
                        self._had_detections = False
                    except Exception:
                        pass

            now = time.time()
            fps = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            # Vẽ overlay
            display_img = self._draw_live_overlay(
                pil_img.copy(), results_cache, had_det_cache, fps)

            self.after(0, lambda img=display_img, r=list(results_cache),
                       hd=had_det_cache, f=fps:
                       self._update_live_ui(img, r, hd, f))
            time.sleep(0.033)

        self.after(0, lambda: self._set_status("Camera đã dừng"))

    def _draw_live_overlay(self, pil_img: Image.Image,
                            results, had_detections, fps):
        """Vẽ overlay lên frame live: bbox nếu detect mode, text nếu classify."""
        if not results:
            return pil_img

        # Nếu có bbox → vẽ bbox
        if had_detections:
            det_only = [r for r in results if r.get("mode") == "detect"]
            pil_img = draw_detection_results(pil_img, det_only, CLASS_COLORS)

        # HUD góc trái: top results
        draw = ImageDraw.Draw(pil_img, "RGBA")
        try:
            font_s = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 13)
            font_b = ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", 15)
        except Exception:
            font_s = font_b = ImageFont.load_default()

        if had_detections:
            # Summarize detections
            label_counts = {}
            for r in results:
                if r.get("mode") == "detect":
                    lbl = r["label"]
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            lines = [f"{lbl}: {cnt}" for lbl, cnt in label_counts.items()]
            title = f"DETECT ({len(results)} obj)"
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
        if self.cls_model is None:
            messagebox.showwarning(
                "Chưa tải model",
                "Model chưa sẵn sàng. Vui lòng đợi trạng thái ● Sẵn sàng.")
            return
        self._set_inferencing(True)
        self._set_process_time(None)
        self._set_status("Đang phân tích ảnh...")
        threading.Thread(target=self._infer_thread,
                         args=(pil_img,), daemon=True).start()

    def _infer_thread(self, pil_img):
        t_start = time.perf_counter()
        try:
            if self.use_detection and self.yolo_model is not None:
                results, had_det = predict_with_detection(
                    self.yolo_model, self.cls_model, self.device,
                    pil_img, self.img_size, self.class_names)
            else:
                results  = predict_cls(
                    self.cls_model, self.device, pil_img,
                    self.img_size, self.class_names, top_k=5)
                had_det = False

            self._last_results   = results
            self._had_detections = had_det

            elapsed = time.perf_counter() - t_start

            # Vẽ bbox lên ảnh nếu detect mode
            if had_det:
                det_only = [r for r in results if r.get("mode") == "detect"]
                annotated = draw_detection_results(pil_img.copy(),
                                                   det_only, CLASS_COLORS)
                elapsed = time.perf_counter() - t_start
                n_obj = len(det_only)
                self.after(0, lambda img=annotated: self._show_pil_on_canvas(img))
                self.after(0, lambda r=results, hd=had_det:
                           self._render_results(r, hd))
                self.after(0, lambda n=n_obj, e=elapsed:
                           self._set_status(f"Phát hiện {n} vật thể"))
            else:
                self.after(0, lambda r=results, hd=had_det:
                           self._render_results(r, hd))
                msg = ("Không phát hiện bbox → phân loại cả ảnh"
                       if self.use_detection else "Phân loại hoàn tất")
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

        if not results:
            self._show_results_empty()
            return

        if had_detections:
            det = [r for r in results if r.get("mode") == "detect"]
            self.lbl_topclass.config(
                text=f"{len(det)} vật thể", fg=PRIMARY)
            for i, r in enumerate(det[:8], 1):
                row = self._make_detect_row(self.result_inner, i, r)
                row.pack(fill="x", pady=4)
                self._result_rows.append(row)
        else:
            top = results[0]
            color = CLASS_COLORS.get(top["label"], ACCENT)
            meta = CLASS_META.get(top["label"], {})
            vi = meta.get("vi", top["label"])
            self.lbl_topclass.config(
                text=f"{vi} · {top['confidence']*100:.1f}%", fg=color)
            for rank, r in enumerate(results, 1):
                row = self._make_cls_row(self.result_inner, rank, r)
                row.pack(fill="x", pady=4)
                self._result_rows.append(row)

        self._result_canvas.configure(
            scrollregion=self._result_canvas.bbox("all"))

    def _make_detect_row(self, parent, rank, r):
        label   = r["label"]
        conf_d  = r["conf_det"]
        conf_c  = r["confidence"]
        bbox    = r["bbox"]
        color   = CLASS_COLORS.get(label, ACCENT)
        meta    = CLASS_META.get(label, {"vi": label})
        is_top  = rank == 1

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