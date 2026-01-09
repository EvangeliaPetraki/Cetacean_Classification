"""
Simplified WhaleNet-style pipeline:
- Load WAV files from class folders
- Resample to a common sample rate
- Center-crop / zero-pad to fixed length (8000 samples)
- Standardize waveforms (zero mean, unit variance)
- Extract Mel spectrogram features on-the-fly
- Train a small ResNet for a small number of epochs + early stopping

This version keeps ONLY the Mel model (simple + strong baseline) and removes:
- class replication (x4 -> 32 labels)
- scattering (WST) models
- ensemble / MLP fusion
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torchaudio
import matplotlib.pyplot as plt

# -----------------------------
# 0) Reproducibility helpers
# -----------------------------
def set_seed(seed: int = 42):
    """Make runs reproducible (as much as possible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 1) Device
# -----------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)


# -----------------------------
# 2) Audio alignment utilities
# -----------------------------
def cutter(X_list, cut_point: int):
    """
    Make all signals exactly 'cut_point' samples long.
    - If shorter: zero-pad equally left/right (centered)
    - If longer: center-crop (keep middle segment)

    Input:
        X_list: list of tensors shaped [1, n_samples] (mono waveforms)
        cut_point: int, target length in samples (e.g., 8000)
    Output:
        Tensor shaped [N, 1, cut_point]
    """
    cut_point = int(cut_point)
    out = []

    for x in X_list:
        n_len = x.shape[1]
        add_pts = cut_point - n_len

        if n_len <= cut_point:
            # pad to the target length (centered)
            pp_left = int(add_pts / 2)
            pp_right = add_pts - pp_left
            out.append(pad(x, (pp_left, pp_right)))
        else:
            # crop to the target length (centered)
            center = int(n_len / 2)
            left = int(cut_point - center)
            right = cut_point - left
            out.append(x[:, center - left : center + right])

    return torch.stack(out, dim=0)  # [N, 1, cut_point]


def standardize_waveforms(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize each waveform independently:
        x <- (x - mean(x)) / (std(x) + eps)

    Input:
        X: Tensor [N, T]  or [N, 1, T]
    Output:
        same shape as input
    """
    if X.ndim == 3:
        # [N, 1, T] -> standardize over time dimension
        mean = X.mean(dim=2, keepdim=True)
        std = X.std(dim=2, keepdim=True)
        return (X - mean) / (std + eps)

    elif X.ndim == 2:
        # [N, T]
        mean = X.mean(dim=1, keepdim=True)
        std = X.std(dim=1, keepdim=True)
        return (X - mean) / (std + eps)

    else:
        raise ValueError("X must be [N,T] or [N,1,T]")


# -----------------------------
# 3) Dataset: Mel Spectrogram on-the-fly
# -----------------------------
class MelSpecDataset(Dataset):
    """
    Dataset that returns (mel_spectrogram, label) pairs.
    - Computes Mel spectrogram inside __getitem__ (on-the-fly)
    - Output is a 2D time-frequency map that we treat as a 1-channel image for CNNs
    """

    def __init__(self, X_wave: torch.Tensor, y: torch.Tensor, sr: int, n_mels: int = 64):
        """
        Inputs:
            X_wave: Tensor [N, T] (waveforms)
            y:      Tensor [N]    (integer labels)
            sr: sample rate used for Mel computation
        """
        self.X = X_wave
        self.y = y
        self.sr = sr

        # MelSpectrogram converts waveform -> Mel-power spectrogram
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            normalized=True,  # consistent with your code
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # waveform expected as [1, T] for torchaudio
        x = self.X[idx].unsqueeze(0)  # [1, T]
        label = self.y[idx]

        # Compute Mel spectrogram: output [n_mels, time_frames]
        m = self.mel(x)

        # CNN expects [C, H, W], so add channel dimension: [1, n_mels, time_frames]
        m = m.unsqueeze(0)

        return m, label


# -----------------------------
# 4) Model: small ResNet
# -----------------------------
class BasicBlock(nn.Module):
    """Standard 2-layer residual block (small ResNet building block)."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)
        return out


class ResNetSmall(nn.Module):
    """
    Small ResNet:
    - Conv -> 3 residual stages -> global avg pool -> linear classifier
    - Stages: [2,2,2] blocks with channels 16, 32, 64
    """

    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        downsample = None

        # If resolution changes (stride != 1) or channel count changes, adjust residual branch
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = [block(self.in_channels, out_channels, stride=stride, downsample=downsample)]
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# 5) Training utilities (few epochs + early stopping)
# -----------------------------
def evaluate(model, loader, criterion):
    """Compute average loss + accuracy on a dataloader."""
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device, dtype=torch.long)

            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)

    return total_loss / n_total, 100.0 * n_correct / n_total


def train_model(
    model,
    train_loader,
    val_loader,
    lr=1e-3,
    weight_decay=1e-3,
    max_epochs=10,           # "few epochs"
    patience=3,              # early stopping patience
    save_path="best_mel_resnet.pt",
):
    """
    Train a model with early stopping based on validation loss.
    - We keep the best checkpoint (lowest validation loss).
    - This is standard practice and makes results more reliable.
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()

        running_loss = 0.0
        n_correct = 0
        n_total = 0

        # ----- Training loop -----
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)

        train_loss = running_loss / n_total
        train_acc = 100.0 * n_correct / n_total

        # ----- Validation loop -----
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"train loss {train_loss:.4f}, train acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f}, val acc {val_acc:.2f}%",
            flush=True
        )

        # ----- Early stopping check -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0

            # Save best model weights
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.", flush=True)
            break

    return history


# -----------------------------
# 6) Main: load data, preprocess, train Mel-ResNet
# -----------------------------
if __name__ == "__main__":
    set_seed(42)

    # Path where the dataset is stored:
    # expected structure:
    # root/
    #   class_1/
    #       file1.wav
    #       file2.wav
    #   class_2/
    #       file3.wav
    dataset_root = "/home/f/fratzeska/E/Cetacean_Classification/_common-frecuent"

    target_sr = 47600          # as used in WhaleNet
    cut_point = 8000           # fixed-length waveform window
    min_per_class = 50         # keep classes with >50 samples (WhaleNet-like)

    # ----- Load WAVs -----
    data_full_list = []   # list of waveforms as torch tensors [1, T]
    y_labels = []         # class name strings

    for root, _, files in os.walk(dataset_root):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            full_path = os.path.join(root, fname)

            # class label = top folder name relative to root
            rel_path = os.path.relpath(full_path, dataset_root)
            class_name = rel_path.split(os.sep)[0]

            try:
                # Read audio as float32; x shape: [samples] for mono, [samples, channels] for stereo
                x, sr = sf.read(full_path, dtype="float32")

                # Convert to torch tensor with shape [channels, samples]
                if x.ndim == 1:
                    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)   # [1, T]
                else:
                    x = torch.tensor(x.T, dtype=torch.float32)              # [C, T]

                # Convert to mono if multi-channel: average across channels
                if x.shape[0] > 1:
                    x = x.mean(dim=0, keepdim=True)

                # Resample if needed
                if sr != target_sr:
                    n_samples = int(x.shape[1] * target_sr / sr)
                    x_rs = resample(x.numpy(), n_samples, axis=1)
                    x = torch.tensor(x_rs, dtype=torch.float32)

                data_full_list.append(x)
                y_labels.append(class_name)

            except Exception as e:
                print(f"Skipping {full_path}: {e}", flush=True)

    print(f"Loaded {len(data_full_list)} wav files.", flush=True)

    # ----- Filter extreme length outliers (optional) -----
    lengths_sec = np.array([x.shape[1] / target_sr for x in data_full_list])
    avg = lengths_sec.mean()
    sd = lengths_sec.std()
    idx = np.where(lengths_sec <= avg + 100 * sd)[0]

    data_sel = [data_full_list[i] for i in idx]
    y_sel = np.array(y_labels)[idx]

    # ----- Align waveforms to fixed length -----
    X_cut = cutter(data_sel, cut_point)          # [N, 1, 8000]

    # ----- Keep only classes with > min_per_class samples -----
    keep_indices = []
    for cname in np.unique(y_sel):
        inds = np.where(y_sel == cname)[0]
        if len(inds) > min_per_class:
            keep_indices.extend(list(inds))

    keep_indices = np.array(sorted(keep_indices))
    X_keep = X_cut[keep_indices]                 # [N_keep, 1, 8000]
    y_keep = y_sel[keep_indices]                 # strings

    print(f"After class filtering: {len(X_keep)} samples", flush=True)

    # ----- Remove exact duplicates (waveform-level) -----
    # Convert to DataFrame for easy duplicate removal (your original approach)
    X_flat = X_keep.squeeze(1).numpy()           # [N, 8000]
    df = pd.DataFrame(X_flat)
    df["y"] = y_keep

    df = df.drop_duplicates(subset=list(range(cut_point)))
    X = torch.tensor(df.iloc[:, :cut_point].values, dtype=torch.float32)   # [N, 8000]
    y = df["y"].values

    # ----- Encode labels to integers 0..(C-1) -----
    le = LabelEncoder()
    y_int = torch.tensor(le.fit_transform(y), dtype=torch.long)
    num_classes = len(le.classes_)
    print(f"Detected {num_classes} classes: {list(le.classes_)}", flush=True)

    # ----- Standardize waveforms (FIXED) -----
    # Your original script computed standardization but then overwrote it.
    X = standardize_waveforms(X)                 # [N, 8000]

    # ----- Train/Validation split (stratified, reproducible) -----
    idx_all = np.arange(len(y_int))
    idx_train, idx_val = train_test_split(
        idx_all,
        test_size=0.25,
        random_state=42,
        stratify=y_int.numpy()
    )

    # ----- Build datasets and dataloaders -----
    batch_size = 64
    train_ds = MelSpecDataset(X[idx_train], y_int[idx_train], sr=target_sr, n_mels=64)
    val_ds   = MelSpecDataset(X[idx_val],   y_int[idx_val],   sr=target_sr, n_mels=64)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ----- Create model -----
    model = ResNetSmall(num_classes=num_classes, in_channels=1).to(device)

    # ----- Train for a few epochs + early stopping -----
    history = train_model(
        model,
        train_loader,
        val_loader,
        lr=1e-3,              # (you used 1e-2; 1e-3 is often more stable for small datasets)
        weight_decay=1e-3,
        max_epochs=10,        # few epochs
        patience=3,           # stop early if val loss stops improving
        save_path="best_mel_resnet.pt"
    )

    print("Training done. Best model saved to best_mel_resnet.pt", flush=True)

    # ----- (Optional) Plot learning curves -----
    plt.figure()
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train acc")
    plt.plot(history["val_acc"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("acc_curve.png", dpi=200)
    plt.close()

    print("Saved loss_curve.png and acc_curve.png", flush=True)
