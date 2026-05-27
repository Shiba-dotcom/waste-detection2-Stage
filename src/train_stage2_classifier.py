# %% [markdown]
# # Stage 2 – EfficientNet-B2 Waste Classifier (5 lớp)
# Huấn luyện EfficientNet-B2 phân loại rác thành 5 nhóm:
# Glass(0), Metal(1), Other(2), Paper(3), Plastic(4).
#
# Pipeline 2 giai đoạn:
#   Stage 1 (YOLO)  → phát hiện vùng chứa rác (binary)
#   Stage 2 (này)   → crop vùng rác → phân loại 5 lớp
#
# Script tương thích Kaggle Notebook (dùng `# %%` cell markers).

# %%
# ============================================================
# Cell 1: Cài đặt thư viện & Import
# ============================================================
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "tqdm"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import io, os, copy, time, json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# Hỗ trợ in tiếng Việt trên mọi console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print(f"[INFO] PyTorch : {torch.__version__}")
print(f"[INFO] timm    : {timm.__version__}")
print(f"[INFO] CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[INFO] GPU     : {torch.cuda.get_device_name(0)}")

# %%
# ============================================================
# Cell 2: Cấu hình đường dẫn & Siêu tham số
# ============================================================
# Chuyển ON_KAGGLE = True khi chạy trên Kaggle Notebook.

ON_KAGGLE = False

if ON_KAGGLE:
    DATA_DIR    = Path("/kaggle/input/waste-classification-merged")
    OUTPUT_DIR  = Path("/kaggle/working/classifier_runs")
else:
    BASE_DIR    = Path(__file__).resolve().parents[1]
    DATA_DIR    = BASE_DIR / "data" / "classification"
    OUTPUT_DIR  = BASE_DIR / "results" / "classifier_runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Tên 5 lớp (thứ tự quan trọng – phải khớp với ImageFolder) -----
CLASS_NAMES = ["Glass", "Metal", "Other", "Paper", "Plastic"]
NUM_CLASSES = len(CLASS_NAMES)

# ----- Siêu tham số -----
IMG_SIZE       = 224          # EfficientNet-B2 input size chuẩn
BATCH_SIZE     = 32
EPOCHS         = 50
LEARNING_RATE  = 1e-4         # AdamW lr – thấp hơn cho fine-tuning pretrained
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
SEED           = 42

# ----- Giới hạn mẫu Plastic (xử lý imbalance) -----
# Plastic thường chiếm đa số → cap lại để tránh model thiên lệch.
PLASTIC_CAP    = 2500

# Đặt seed cho reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")
print(f"[INFO] Data  : {DATA_DIR}")

# %%
# ============================================================
# Cell 3: Định nghĩa Data Augmentation
# ============================================================
# Train: augmentation mạnh để tăng tính tổng quát.
#   - RandomResizedCrop: crop ngẫu nhiên, buộc model học nhiều tỷ lệ.
#   - ColorJitter: thay đổi ánh sáng, tương phản – mô phỏng điều kiện thực tế.
#   - RandomRotation(15): xoay nhẹ, rác có thể nằm ở mọi góc.
# Val/Test: chỉ resize + center crop, KHÔNG augment (đánh giá công bằng).
# Normalize: dùng ImageNet mean/std vì model pretrained trên ImageNet.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# %%
# ============================================================
# Cell 4: Tải dữ liệu & Xử lý mất cân bằng (Imbalance)
# ============================================================
# Sử dụng 3 kỹ thuật đồng thời để xử lý class imbalance:
#   1. Cap mẫu Plastic ≤ PLASTIC_CAP (giảm lớp đa số)
#   2. WeightedRandomSampler (lấy mẫu cân bằng trong mỗi batch)
#   3. CrossEntropyLoss(weight=...) (phạt nặng hơn khi sai lớp thiểu số)


def cap_class_samples(dataset, class_name, max_samples, class_names, seed=42):
    """
    Giới hạn số mẫu của một lớp cụ thể trong dataset.

    Tham số:
        dataset     : torchvision.datasets.ImageFolder
        class_name  : tên lớp cần cap (ví dụ: "Plastic")
        max_samples : số mẫu tối đa
        class_names : danh sách tên lớp theo thứ tự class_to_idx
        seed        : seed cho random sampling

    Trả về:
        Subset chứa chỉ mục đã được giới hạn.
    """
    # Tìm class_id từ tên lớp
    class_id = dataset.class_to_idx[class_name]

    # Phân loại chỉ mục theo lớp
    rng = np.random.RandomState(seed)
    keep_indices = []

    for idx, (_, label) in enumerate(dataset.samples):
        if label == class_id:
            keep_indices.append((idx, True))   # đánh dấu là lớp cần cap
        else:
            keep_indices.append((idx, False))

    # Tách chỉ mục lớp cần cap vs các lớp khác
    target_indices = [idx for idx, is_target in keep_indices if is_target]
    other_indices  = [idx for idx, is_target in keep_indices if not is_target]

    # Cap nếu vượt quá giới hạn
    if len(target_indices) > max_samples:
        target_indices = rng.choice(target_indices, size=max_samples, replace=False).tolist()
        print(f"  [CAP] {class_name}: giới hạn còn {max_samples} mẫu "
              f"(giảm {len([i for i, t in keep_indices if t]) - max_samples} mẫu)")

    final_indices = sorted(other_indices + target_indices)
    return Subset(dataset, final_indices), final_indices


def compute_class_distribution(dataset, indices=None):
    """
    Đếm số mẫu mỗi lớp, trả về Counter {class_id: count}.
    Nếu indices=None, đếm toàn bộ dataset.
    """
    if indices is not None:
        targets = [dataset.targets[i] for i in indices]
    else:
        targets = dataset.targets
    return Counter(targets)


# --- Tải dataset ---
print("=" * 60)
print("  LOADING DATASET")
print("=" * 60)

# ImageFolder tự động gán label theo tên thư mục (alphabetical order).
# Thứ tự: Glass=0, Metal=1, Other=2, Paper=3, Plastic=4 → khớp CLASS_NAMES.
train_dataset_full = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_transforms)
val_dataset   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=eval_transforms)
test_dataset  = datasets.ImageFolder(str(DATA_DIR / "test"),  transform=eval_transforms)

# Xác nhận thứ tự lớp khớp với CLASS_NAMES
actual_classes = list(train_dataset_full.class_to_idx.keys())
print(f"  Lớp phát hiện : {actual_classes}")
assert actual_classes == CLASS_NAMES, (
    f"[LỖI] Thứ tự lớp không khớp!\n"
    f"  Mong đợi : {CLASS_NAMES}\n"
    f"  Thực tế  : {actual_classes}"
)

# --- Kỹ thuật 1: Cap mẫu Plastic ---
print(f"\n  [1/3] Cap Plastic ≤ {PLASTIC_CAP} mẫu")
dist_before = compute_class_distribution(train_dataset_full)
print(f"  Phân bố TRƯỚC khi cap:")
for cls_id, name in enumerate(CLASS_NAMES):
    print(f"    {name:>10s}: {dist_before[cls_id]:>5,}")

train_dataset, train_indices = cap_class_samples(
    train_dataset_full, "Plastic", PLASTIC_CAP, CLASS_NAMES, seed=SEED
)

dist_after = compute_class_distribution(train_dataset_full, train_indices)
print(f"\n  Phân bố SAU khi cap:")
for cls_id, name in enumerate(CLASS_NAMES):
    print(f"    {name:>10s}: {dist_after[cls_id]:>5,}")

# --- Kỹ thuật 2: WeightedRandomSampler ---
# Mỗi mẫu có trọng số = 1 / (số mẫu trong lớp đó).
# → Lớp ít mẫu sẽ được sample thường xuyên hơn mỗi epoch.
print(f"\n  [2/3] Tạo WeightedRandomSampler")

train_targets = [train_dataset_full.targets[i] for i in train_indices]
class_counts  = np.array([dist_after[c] for c in range(NUM_CLASSES)], dtype=np.float64)

# Trọng số cho mỗi mẫu = 1 / count(lớp của mẫu đó)
sample_weights = np.array([1.0 / class_counts[t] for t in train_targets])
sample_weights = torch.from_numpy(sample_weights).double()

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),       # Sample đủ 1 epoch
    replacement = True,                      # Phải True cho WeightedRandomSampler
)

print(f"  Số mẫu mỗi epoch: {len(sample_weights):,}")

# --- Kỹ thuật 3: Class weights cho Loss function ---
# Inverse frequency: lớp ít mẫu → weight cao → loss phạt nặng hơn khi sai.
print(f"\n  [3/3] Tính class weights cho CrossEntropyLoss")

total_samples = sum(class_counts)
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)

for cls_id, name in enumerate(CLASS_NAMES):
    print(f"    {name:>10s}: weight = {class_weights[cls_id]:.4f}  (n={int(class_counts[cls_id]):,})")

# --- Tạo DataLoader ---
train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    sampler     = sampler,          # Dùng sampler → KHÔNG set shuffle
    num_workers = NUM_WORKERS,
    pin_memory  = True,
    drop_last   = True,             # Bỏ batch cuối nếu không đủ → ổn định BatchNorm
)

val_loader = DataLoader(
    val_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = NUM_WORKERS,
    pin_memory  = True,
)

print(f"\n  Train batches/epoch : {len(train_loader)}")
print(f"  Val   batches       : {len(val_loader)}")
print(f"  Test  batches       : {len(test_loader)}")

# %%
# ============================================================
# Cell 5: Xây dựng Model – EfficientNet-B2
# ============================================================
# Sử dụng timm để load EfficientNet-B2 pretrained trên ImageNet.
# Thay thế classifier head cuối cùng cho 5 lớp rác.
# EfficientNet-B2 có 9.1M params, cân bằng giữa accuracy và tốc độ.

model = timm.create_model("efficientnet_b2", pretrained=True)

# Lấy số features của lớp cuối cùng để thay thế
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, NUM_CLASSES)

model = model.to(DEVICE)

# Tổng số params
total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[INFO] EfficientNet-B2")
print(f"  Tổng params      : {total_params:>12,}")
print(f"  Trainable params : {trainable_params:>12,}")
print(f"  Classifier head  : {in_features} → {NUM_CLASSES}")

# %%
# ============================================================
# Cell 6: Loss, Optimizer, Scheduler
# ============================================================
# CrossEntropyLoss có class weights → phạt nặng hơn khi sai lớp thiểu số.
# AdamW: Adam + weight decay decoupled – tốt cho fine-tuning.
# CosineAnnealingLR: lr giảm theo hình cos, giúp thoát local minima.

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Mixed precision – tăng tốc 1.5-2x trên GPU (T4, P100, V100...)
# GradScaler giúp tránh underflow khi dùng FP16.
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

print(f"[INFO] Loss      : CrossEntropyLoss (weighted)")
print(f"[INFO] Optimizer : AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"[INFO] Scheduler : CosineAnnealingLR (T_max={EPOCHS})")
print(f"[INFO] AMP       : {'Enabled' if torch.cuda.is_available() else 'Disabled (CPU)'}")

# %%
# ============================================================
# Cell 7: Training Loop
# ============================================================
# Vòng lặp huấn luyện chính:
#   - Mỗi epoch: train → validate → log metrics → save best
#   - Mixed precision (AMP) cho tốc độ
#   - tqdm progress bar cho mỗi batch
#   - Track best model theo val accuracy


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Huấn luyện 1 epoch, trả về (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass với mixed precision
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Thống kê
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Cập nhật progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Đánh giá model, trả về (avg_loss, accuracy, all_preds, all_labels)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=100)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


# --------- Main Training Loop ---------
print("\n" + "=" * 70)
print("  BẮT ĐẦU HUẤN LUYỆN")
print("=" * 70)

best_val_acc  = 0.0
best_epoch    = 0
best_model_wts = copy.deepcopy(model.state_dict())

# Lưu lịch sử metrics để vẽ biểu đồ
history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [],
    "lr": [],
}

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()
    current_lr  = optimizer.param_groups[0]["lr"]

    print(f"\nEpoch {epoch}/{EPOCHS}  (lr={current_lr:.2e})")
    print("-" * 50)

    # --- Train ---
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE
    )

    # --- Validate ---
    val_loss, val_acc, _, _ = evaluate(
        model, val_loader, criterion, DEVICE
    )

    # --- Scheduler step ---
    scheduler.step()

    # --- Log ---
    epoch_time = time.time() - epoch_start
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"  Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f}  |  Val   Acc: {val_acc:.4f}")
    print(f"  Time: {epoch_time:.1f}s")

    # --- Save best model ---
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_epoch    = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  ★ New best! Val Acc = {val_acc:.4f}")

total_time = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"  HOÀN TẤT HUẤN LUYỆN")
print(f"  Tổng thời gian : {total_time/60:.1f} phút")
print(f"  Best epoch      : {best_epoch}")
print(f"  Best val acc    : {best_val_acc:.4f}")
print(f"{'=' * 70}")

# %%
# ============================================================
# Cell 8: Vẽ biểu đồ Training Curves
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("EfficientNet-B2 Classifier – Training Curves", fontsize=14, fontweight="bold")
epochs_range = range(1, EPOCHS + 1)

# --- Loss ---
ax = axes[0]
ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="tab:blue")
ax.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="tab:orange")
ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best (epoch {best_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Loss")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Accuracy ---
ax = axes[1]
ax.plot(epochs_range, history["train_acc"], label="Train Acc", color="tab:blue")
ax.plot(epochs_range, history["val_acc"],   label="Val Acc",   color="tab:orange")
ax.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best (epoch {best_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy")
ax.legend()
ax.grid(True, alpha=0.3)

# --- Learning Rate ---
ax = axes[2]
ax.plot(epochs_range, history["lr"], label="Learning Rate", color="tab:green")
ax.set_xlabel("Epoch")
ax.set_ylabel("LR")
ax.set_title("Learning Rate Schedule")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = OUTPUT_DIR / "training_curves.png"
plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
print(f"[INFO] Đã lưu biểu đồ: {chart_path}")
plt.show()

# %%
# ============================================================
# Cell 9: Đánh giá trên tập Test – Per-class Metrics
# ============================================================
# Load best model → chạy trên test set → tính precision, recall, f1 cho mỗi lớp.
# Confusion matrix giúp xác định lớp nào hay bị nhầm lẫn.

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load best weights
model.load_state_dict(best_model_wts)
model.eval()

# Chạy inference trên test set
test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, DEVICE
)

print("\n" + "=" * 70)
print("  KẾT QUẢ TRÊN TẬP TEST")
print("=" * 70)
print(f"  Test Loss     : {test_loss:.4f}")
print(f"  Test Accuracy : {test_acc:.4f}")
print("=" * 70)

# --- Per-class metrics ---
print("\n  CHI TIẾT TỪNG LỚP:")
print("-" * 70)
report = classification_report(
    test_labels, test_preds,
    target_names=CLASS_NAMES,
    digits=4,
)
print(report)

# --- Confusion Matrix ---
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(8, 7))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix – Test Set", fontsize=13, fontweight="bold")
plt.tight_layout()

cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(str(cm_path), dpi=150, bbox_inches="tight")
print(f"[INFO] Đã lưu confusion matrix: {cm_path}")
plt.show()

# %%
# ============================================================
# Cell 10: Lưu model weights
# ============================================================
# Lưu best weights ở 2 nơi:
#   1. Thư mục output (classifier_runs/) – cho phân tích
#   2. Thư mục models/ – cho pipeline inference chính

import shutil

# Lưu vào output dir
weights_path = OUTPUT_DIR / "stage2_efficientnet_b2_best.pth"
torch.save({
    "model_state_dict": best_model_wts,
    "class_names":      CLASS_NAMES,
    "num_classes":      NUM_CLASSES,
    "img_size":         IMG_SIZE,
    "best_epoch":       best_epoch,
    "best_val_acc":     best_val_acc,
    "test_acc":         test_acc,
    "history":          history,
}, str(weights_path))
print(f"[INFO] Đã lưu checkpoint: {weights_path}")

# Copy sang models/ để pipeline inference dùng
if ON_KAGGLE:
    final_dst = Path("/kaggle/working/stage2_best.pth")
else:
    final_dst = Path(__file__).resolve().parents[1] / "models" / "stage2_best.pth"

final_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(str(weights_path), str(final_dst))
print(f"[INFO] Đã copy weights → {final_dst}")

# Lưu history ra JSON để phân tích sau
history_path = OUTPUT_DIR / "training_history.json"
with open(str(history_path), "w", encoding="utf-8") as f:
    json.dump(history, f, indent=2)
print(f"[INFO] Đã lưu history: {history_path}")

# %%
# ============================================================
# Cell 11: Tổng kết
# ============================================================
print("\n" + "=" * 70)
print("  HOÀN TẤT HUẤN LUYỆN STAGE 2 – CLASSIFIER")
print("=" * 70)
print(f"  Model           : EfficientNet-B2 (5 lớp)")
print(f"  Lớp             : {', '.join(CLASS_NAMES)}")
print(f"  Best epoch      : {best_epoch}/{EPOCHS}")
print(f"  Best val acc    : {best_val_acc:.4f}")
print(f"  Test accuracy   : {test_acc:.4f}")
print(f"  Weights         : {final_dst}")
print(f"  Training curves : {chart_path}")
print(f"  Confusion matrix: {cm_path}")
print("=" * 70)
print("\n[DONE] Pipeline 2 giai đoạn hoàn tất!")
print("  Stage 1: YOLO binary detector  → phát hiện vùng rác")
print("  Stage 2: EfficientNet-B2       → phân loại 5 lớp rác")
print("  → Sẵn sàng ghép vào pipeline inference.")
