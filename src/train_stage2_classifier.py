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

# Tự động phát hiện môi trường, hoặc đọc biến môi trường ON_KAGGLE=1
import os as _os
ON_KAGGLE = _os.environ.get("ON_KAGGLE", "0") == "1" or _os.path.exists("/kaggle/working")

if ON_KAGGLE:
    DATA_DIR    = Path("/kaggle/working/waste-detection2-Stage/data/classification_merged")
    OUTPUT_DIR  = Path("/kaggle/working/waste-detection2-Stage/results/classifier_runs")
else:
    BASE_DIR    = Path(__file__).resolve().parents[1]
    DATA_DIR    = BASE_DIR / "data" / "classification_merged"  # Dùng tập đã trộn (có TrashNet, RealWaste & Background)
    OUTPUT_DIR  = BASE_DIR / "results" / "classifier_runs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Tên 6 lớp (thứ tự quan trọng – phải khớp với ImageFolder chữ cái đầu) -----
CLASS_NAMES = ["Background", "Glass", "Metal", "Other", "Paper", "Plastic"]
NUM_CLASSES = len(CLASS_NAMES)

# ----- Siêu tham số -----
IMG_SIZE       = 224          # EfficientNet-B2 input size chuẩn
BATCH_SIZE     = 32
EPOCHS         = 60           # Tăng lên để bù cho 2 giai đoạn train
NUM_WORKERS    = 4
SEED           = 42

# ----- 2-Phase Training (chống Overfitting) -----
# Giai đoạn 1: Freeze backbone → chỉ train head (nhanh, ổn định)
# Giai đoạn 2: Unfreeze toàn bộ → fine-tune với LR thấp hơn
PHASE1_EPOCHS  = 10           # Số epoch train với backbone đóng băng
PHASE1_LR      = 5e-4         # LR cao hơn khi chỉ train head
PHASE2_LR      = 5e-5         # LR rất thấp khi fine-tune toàn bộ
WEIGHT_DECAY   = 3e-4         # Tăng L2 regularization (gốc: 1e-4)
DROPOUT_RATE   = 0.4          # Dropout trước classifier head
LABEL_SMOOTHING = 0.1         # Tránh model quá tự tin
EARLY_STOP_PATIENCE = 12      # Dừng sớm nếu val_loss không cải thiện

# ----- Giới hạn mẫu Plastic (xử lý imbalance) -----
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
# Cell 3: Định nghĩa Data Augmentation (Tăng cường chống Overfitting)
# ============================================================
# Augmentation mạnh hơn để buộc model học feature tổng quát:
#   - RandAugment: tự động chọn & kết hợp phép augment tối ưu
#   - RandomErasing: xóa ngẫu nhiên 1 vùng ảnh → tránh học vị trí cố định
#   - RandomVerticalFlip: rác nằm ở mọi hướng trong thực tế
#   - scale=(0.6, 1.0): crop aggressively hơn (gốc: 0.8-1.0)
# Val/Test: chỉ resize + center crop, KHÔNG augment.

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),  # Crop aggressively hơn
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),                      # [MỚI] Rác có thể lật dọc
    transforms.ColorJitter(brightness=0.4, contrast=0.4,       # Mạnh hơn (gốc: 0.3)
                           saturation=0.3, hue=0.1),
    transforms.RandomRotation(30),                             # Xoay rộng hơn (gốc: 15°)
    transforms.RandAugment(num_ops=2, magnitude=9),            # [MỚI] Auto augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),        # [MỚI] Xóa vùng ngẫu nhiên
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

print("[INFO] Augmentation đã được tăng cường:")
print("  + RandAugment(num_ops=2, magnitude=9)")
print("  + RandomVerticalFlip(p=0.2)")
print("  + RandomErasing(p=0.3)")
print("  + ColorJitter mạnh hơn (brightness/contrast 0.4)")
print("  + RandomRotation(30°)")

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
# Cell 5: Xây dựng Model – EfficientNet-B2 + Dropout
# ============================================================
# Cải tiến so với phiên bản gốc:
#   1. Thêm Dropout(p=0.4) trước classifier head → L2 regularization
#   2. Backbone sẽ bị đóng băng ở Giai đoạn 1 (Phase 1)

model = timm.create_model("efficientnet_b2", pretrained=True)

# Thay classifier head: Linear → Dropout + Linear
in_features = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),          # [MỚI] Dropout chống overfitting
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

# Đóng băng backbone cho Giai đoạn 1
def freeze_backbone(model):
    """Đóng băng toàn bộ trừ classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_all(model):
    """Mở khóa toàn bộ model."""
    for param in model.parameters():
        param.requires_grad = True

freeze_backbone(model)  # Bắt đầu với backbone bị đóng băng

# Thống kê params
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params    = total_params - trainable_params

print(f"[INFO] EfficientNet-B2 (Phase 1 – Frozen backbone)")
print(f"  Tổng params      : {total_params:>12,}")
print(f"  Trainable params : {trainable_params:>12,}  (chỉ classifier head)")
print(f"  Frozen params    : {frozen_params:>12,}  (backbone)")
print(f"  Classifier head  : Dropout({DROPOUT_RATE}) → Linear({in_features} → {NUM_CLASSES})")

# %%
# ============================================================

# Mixed precision – tăng tốc 1.5-2x trên GPU (T4, P100, V100...)
# GradScaler giúp tránh underflow khi dùng FP16.
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

print(f"[INFO] Loss      : CrossEntropyLoss (weighted)")
print(f"[INFO] Optimizer : AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"[INFO] Scheduler : CosineAnnealingLR (T_max={EPOCHS})")
print(f"[INFO] AMP       : {'Enabled' if torch.cuda.is_available() else 'Disabled (CPU)'}")

# %%
# ============================================================
# Cell 7: Training Loop – 2 Giai đoạn + Early Stopping
# ============================================================
# GIAI ĐOẠN 1 (Phase 1): Backbone FROZEN, chỉ train head
#   → LR cao hơn, hội tụ nhanh, không phá vỡ pretrained weights
# GIAI ĐOẠN 2 (Phase 2): Unfreeze toàn bộ, fine-tune với LR rất thấp
#   → Tinh chỉnh toàn bộ model theo domain rác
# EARLY STOPPING: dừng nếu val_loss không cải thiện sau N epoch


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

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    return running_loss / total, correct / total


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

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ─────────────── Main Training Loop ───────────────
print("\n" + "=" * 70)
print(f"  BẮT ĐẦU HUẤN LUYỆN 2 GIAI ĐOẠN")
print(f"  Phase 1: epoch  1 → {PHASE1_EPOCHS} (backbone FROZEN,   lr={PHASE1_LR:.0e})")
print(f"  Phase 2: epoch {PHASE1_EPOCHS+1} → {EPOCHS}  (backbone UNFROZEN, lr={PHASE2_LR:.0e})")
print("=" * 70)

best_val_acc   = 0.0
best_val_loss  = float("inf")
best_epoch     = 0
best_model_wts = copy.deepcopy(model.state_dict())
early_stop_counter = 0
phase = 1

history = {
    "train_loss": [], "train_acc": [],
    "val_loss":   [], "val_acc":   [],
    "lr": [], "phase": [],
}

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # ── Chuyển sang Phase 2 ──
    if epoch == PHASE1_EPOCHS + 1 and phase == 1:
        phase = 2
        print(f"\n{'━'*70}")
        print(f"  🔓 PHASE 2: Unfreeze toàn bộ backbone – Fine-tune với lr={PHASE2_LR:.0e}")
        print(f"{'━'*70}")
        unfreeze_all(model)
        # Reset optimizer với LR thấp hơn cho toàn bộ params
        optimizer = optim.AdamW(
            model.parameters(),
            lr=PHASE2_LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(EPOCHS - PHASE1_EPOCHS)
        )
        early_stop_counter = 0  # Reset Early Stopping khi chuyển phase
        best_val_loss = float("inf")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable params: {trainable:,}")

    current_lr = optimizer.param_groups[0]["lr"]
    phase_label = "🧊 P1" if phase == 1 else "🔥 P2"
    print(f"\nEpoch {epoch:>3}/{EPOCHS}  {phase_label}  lr={current_lr:.2e}")
    print("-" * 50)

    # ── Train & Validate ──
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE
    )
    val_loss, val_acc, _, _ = evaluate(
        model, val_loader, criterion, DEVICE
    )
    scheduler.step()

    epoch_time = time.time() - epoch_start

    # ── Log ──
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)
    history["phase"].append(phase)

    gap = train_acc - val_acc
    print(f"  Train: loss={train_loss:.4f}  acc={train_acc:.4f}")
    print(f"  Val  : loss={val_loss:.4f}  acc={val_acc:.4f}  (gap={gap:+.4f})")
    print(f"  Time : {epoch_time:.1f}s")

    # ── Lưu best model (theo val_acc) ──
    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        best_epoch     = epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        print(f"  ★ New best! Val Acc = {val_acc:.4f}")

    # ── Early Stopping (theo val_loss) ──
    if val_loss < best_val_loss - 1e-4:   # Cải thiện đáng kể
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        print(f"  ⏳ Early stop counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")
        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  🛑 Early Stopping! Val loss không cải thiện sau {EARLY_STOP_PATIENCE} epoch.")
            print(f"     Dừng tại epoch {epoch}, best model tại epoch {best_epoch}.")
            break

total_time = time.time() - start_time
print(f"\n{'=' * 70}")
print(f"  HOÀN TẤT HUẤN LUYỆN")
print(f"  Tổng thời gian  : {total_time/60:.1f} phút")
print(f"  Best epoch      : {best_epoch}")
print(f"  Best val acc    : {best_val_acc:.4f}")
print(f"  Kết thúc epoch  : {epoch}")
print(f"{'=' * 70}")

# %%
# ============================================================
# Cell 8: Vẽ biểu đồ Training Curves (có đánh dấu Phase 1/2)
# ============================================================

actual_epochs = len(history["train_loss"])
epochs_range  = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("EfficientNet-B2 Classifier – Training Curves (2-Phase + Early Stopping)",
             fontsize=13, fontweight="bold")

# Đường phân cách Phase 1 / Phase 2
def mark_phase(ax):
    if PHASE1_EPOCHS < actual_epochs:
        ax.axvline(x=PHASE1_EPOCHS, color="purple", linestyle=":",
                   alpha=0.7, linewidth=1.5, label=f"Phase 2 bắt đầu (epoch {PHASE1_EPOCHS})")
    ax.axvline(x=best_epoch, color="red", linestyle="--",
               alpha=0.6, linewidth=1.5, label=f"Best (epoch {best_epoch})")

# --- Loss ---
ax = axes[0]
ax.plot(epochs_range, history["train_loss"], label="Train Loss", color="tab:blue")
ax.plot(epochs_range, history["val_loss"],   label="Val Loss",   color="tab:orange")
mark_phase(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- Accuracy + Overfitting Gap ---
ax = axes[1]
ax.plot(epochs_range, history["train_acc"], label="Train Acc", color="tab:blue")
ax.plot(epochs_range, history["val_acc"],   label="Val Acc",   color="tab:orange")
# Tô màu vùng gap (overfitting)
ax.fill_between(epochs_range, history["val_acc"], history["train_acc"],
                alpha=0.15, color="red", label="Overfitting gap")
mark_phase(ax)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Accuracy & Overfitting Gap")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)

# --- Learning Rate ---
ax = axes[2]
ax.plot(epochs_range, history["lr"], color="tab:green")
if PHASE1_EPOCHS < actual_epochs:
    ax.axvline(x=PHASE1_EPOCHS, color="purple", linestyle=":", alpha=0.7,
               label=f"Phase 2 (lr reset → {PHASE2_LR:.0e})")
ax.set_xlabel("Epoch"); ax.set_ylabel("LR"); ax.set_title("Learning Rate Schedule")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

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
gap_final = history["train_acc"][-1] - history["val_acc"][-1]
gap_best  = history["train_acc"][best_epoch-1] - history["val_acc"][best_epoch-1]

print("\n" + "=" * 70)
print("  HOÀN TẤT HUẤN LUYỆN STAGE 2 – CLASSIFIER (Cải tiến)")
print("=" * 70)
print(f"  Model            : EfficientNet-B2 (5 lớp)")
print(f"  Lớp              : {', '.join(CLASS_NAMES)}")
print(f"  Kỹ thuật mới     : Freeze→Unfreeze, Dropout({DROPOUT_RATE}), RandAugment,")
print(f"                     LabelSmoothing({LABEL_SMOOTHING}), EarlyStopping")
print(f"  Best epoch       : {best_epoch} (/{len(history['train_loss'])} epoch thực tế)")
print(f"  Best val acc     : {best_val_acc:.4f}")
print(f"  Test accuracy    : {test_acc:.4f}")
print(f"  Overfitting gap  : {gap_best:+.4f} (tại best epoch)")
print(f"  Weights          : {final_dst}")
print(f"  Training curves  : {chart_path}")
print(f"  Confusion matrix : {cm_path}")
print("=" * 70)
print("\n  So sánh với phiên bản gốc (nếu có cải thiện):")
print(f"  Gốc  → Best val acc = 70.1% (epoch 6/50, gap ≈ 30%)")
print(f"  Mới  → Best val acc = {best_val_acc:.1%} (epoch {best_epoch}, gap ≈ {gap_best:.1%})")
print("\n[DONE] Pipeline 2 giai đoạn hoàn tất!")
print("  Stage 1: YOLO binary detector  → phát hiện vùng rác")
print("  Stage 2: EfficientNet-B2       → phân loại 6 lớp (có Background để lọc FP)")
print("  → Sẵn sàng ghép vào pipeline inference.")
