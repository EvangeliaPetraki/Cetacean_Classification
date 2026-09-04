#!/usr/bin/env python3
"""
whale_ensembles.py  --  LORO (leave-one-recording-out) variant.

Script for the HPC.

Difference from the root-project version: the train/test split is done by SOURCE
RECORDING, not by clip. Every clip of one Watkins recording (filename-prefix key)
lands entirely inside one CV fold's held-out set, so the reported accuracy
estimates performance on *unseen recordings* instead of memorised recording
conditions (background, hydrophone, session).

What it does:
1) Load WAV files from <Species>/<Year>/<ID>.wav, deriving a source-recording key
   per clip alongside X and y (kept aligned through dedup + class filtering).
2) Resample to target SR, convert to mono.
3) Center-crop or symmetric zero-pad to fixed length (T samples).
4) Standardize each waveform independently (zero mean, unit variance).
5) Build recording-grouped CV folds:
     --cv loro        -> LeaveOneGroupOut  (one fold per recording)
     --cv group-kfold -> StratifiedGroupKFold(--n_splits)
   Each fold carves a grouped early-stopping validation set from its training
   recordings (--val_frac), so val is leakage-safe too.
6) Extract features:
   - Mel spectrogram (on-the-fly per batch)
   - WST order-1 coefficients (computed once per (J,Q), reused across folds+configs)
7) Train lightweight models per fold: ResNet-small, TinyCNN, MobileNetV3-small,
   EfficientNet-B0.
8) Stitch per-fold predictions into one out-of-fold (OOF) prediction per clip and
   score once: metrics.json, classification_report_test.txt, per_class_metrics_test.*,
   confusion_test.png, oof_predictions.npz, per_recording_accuracy.csv, folds_summary.csv.
9) Resume safely: a fold with folds/fold_XXX/predictions.npz is reused, not retrained.

Ensembles (--ensemble_runs): soft-vote over the members' saved OOF predictions
(members must share the same folds.json).
"""

import os
import json
import time
import math
import random
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import pad
import torch.nn.functional as F
import torchaudio

from sklearn.model_selection import (
    train_test_split,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   #
import matplotlib.pyplot as plt

# -----------------------------
#   LORO output namespacing
# -----------------------------
# Every artefact this (LORO) pipeline writes -- run/ensemble directory names,
# metrics, reports, confusion figures, summaries, the preprocessing cache -- is
# prefixed so it is self-identifying and can never overwrite the pre-LORO
# pipeline's files. The per-fold artefacts written by the (deliberately
# untouched) train_one_run live under <loro_run>/folds/fold_XXX/ and keep their
# original names; their path already makes their origin unambiguous.
LORO_PREFIX = "loro_"


def _lp(name: str) -> str:
    """Prefix an output basename with LORO_PREFIX (idempotent)."""
    return name if name.startswith(LORO_PREFIX) else f"{LORO_PREFIX}{name}"


# -----------------------------
#   Seeds
# -----------------------------
def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN makes results more reproducible on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# 1) Device
# -----------------------------
def get_device():
    """Use GPU if available."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 2) Audio preprocessing
# -----------------------------
def center_crop_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Ensure waveform length is exactly target_len samples.
    Input:  x [1, T] (mono)
    Output: x [1, target_len]
    """
    T = x.shape[1]
    if T == target_len:
        return x

    if T < target_len:
        # symmetric zero-pad
        add = target_len - T
        left = add // 2
        right = add - left
        return pad(x, (left, right))

    # center-crop
    center = T // 2
    half = target_len // 2
    start = max(0, center - half)
    end = start + target_len
    return x[:, start:end]


def standardize_per_signal(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Standardize each waveform independently:
        x <- (x - mean(x)) / (std(x) + eps)
    Input:  X [N, T]
    Output: X [N, T]
    """
    mean = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True)
    return (X - mean) / (std + eps)


def recording_id_from_relpath(rel: str, id_len: int = 5) -> str:
    """
    Source-recording key for a clip, from its path relative to dataset_root.

    Watkins layout: <Species>/<Year>/<ID>.wav , where <ID> is an <id_len>-char
    master-recording code followed by a per-clip serial (e.g. 61027001, 61027002,
    ... = clips 1,2,... of Sperm-Whale recording 61027). Grouping by this key is
    what keeps every clip of one recording on the same side of a CV split, so the
    model can't score by memorising a recording's channel/background signature.

    Key = "<Species>/<sub-dirs>/<ID[:id_len]>"  (sub-dirs = the year folder, etc.)
    """
    parts = rel.split(os.sep)
    species = parts[0]
    subdirs = parts[1:-1]
    stem = os.path.splitext(parts[-1])[0]
    prefix = stem[:id_len] if len(stem) >= id_len else stem
    return "/".join([species, *subdirs, prefix])


def load_and_preprocess_dataset(
    dataset_root: str,
    target_sr: int,
    target_len: int,
    min_per_class: int,
    cache_path: str,
    recording_id_len: int = 5,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Loads WAV files and applies preprocessing:
    - read wav
    - convert to mono
    - resample
    - center-crop/pad to target_len
    - remove exact duplicates
    - keep classes with > min_per_class samples
    - standardize waveforms

    Returns:
        X:      torch.Tensor [N, T] float32
        y:      np.ndarray of class strings, length N
        groups: np.ndarray of source-recording keys, length N (aligned to X, y)
    """

    # If cache exists, load it and return immediately.
    if os.path.exists(cache_path):
        print(f"[Cache] Loading preprocessed data from: {cache_path}", flush=True)
        ckpt = torch.load(cache_path, map_location="cpu",  weights_only=False)
        if "groups" not in ckpt:
            raise RuntimeError(
                f"{cache_path} was written by the pre-LORO pipeline and has no 'groups' "
                "array. Delete it and re-run so the grouped cache is rebuilt."
            )
        return ckpt["X"], ckpt["y"], ckpt["groups"]

    print("[Data] Loading wav files...", flush=True)

    waveforms: List[torch.Tensor] = []
    labels: List[str] = []
    rec_ids: List[str] = []

    # Expect dataset structure:
    # root/
    #   classA/.../*.wav
    #   classB/.../*.wav
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            full_path = os.path.join(root, fname)
            rel = os.path.relpath(full_path, dataset_root)
            class_name = rel.split(os.sep)[0]

            try:
                x, sr = sf.read(full_path, dtype="float32")

                # Convert to torch tensor [channels, samples]
                if x.ndim == 1:
                    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, T]
                else:
                    x = torch.tensor(x.T, dtype=torch.float32)             # [C, T]

                # Convert to mono if multi-channel
                if x.shape[0] > 1:
                    x = x.mean(dim=0, keepdim=True)

                # Resample to common sampling rate
                if sr != target_sr:
                    new_n = int(x.shape[1] * target_sr / sr)
                    x_rs = resample(x.numpy(), new_n, axis=1)
                    x = torch.tensor(x_rs, dtype=torch.float32)

                # Align length
                x = center_crop_or_pad(x, target_len)

                waveforms.append(x)
                labels.append(class_name)
                rec_ids.append(recording_id_from_relpath(rel, recording_id_len))

            except Exception as e:
                print(f"[Warn] Skipping {full_path}: {e}", flush=True)

    print(f"[Data] Loaded {len(waveforms)} waveforms", flush=True)

    # Stack into array [N, 1, T] then squeeze -> [N, T]
    X = torch.stack(waveforms, dim=0).squeeze(1)  # [N, T]
    y = np.array(labels)
    g = np.array(rec_ids)

    # Remove exact duplicates in waveform samples (y and groups ride along so
    # they stay aligned to the surviving rows).
    df = pd.DataFrame(X.numpy())
    df["y"] = y
    df["g"] = g
    df = df.drop_duplicates(subset=list(range(target_len)))
    X = torch.tensor(df.iloc[:, :target_len].values, dtype=torch.float32)
    y = df["y"].values
    g = df["g"].values

    # Keep only classes with enough samples
    keep = []
    for cname in np.unique(y):
        idx = np.where(y == cname)[0]
        if len(idx) > min_per_class:
            keep.extend(idx.tolist())
    keep = np.array(sorted(keep))

    X = X[keep]
    y = y[keep]
    g = g[keep]

    print(f"[Data] After class filtering: {len(X)} samples, "
          f"{len(np.unique(g))} source recordings", flush=True)

    # Standardize per waveform (important for stable training)
    X = standardize_per_signal(X)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"X": X, "y": y, "groups": g}, cache_path)
    print(f"[Cache] Saved preprocessed data to: {cache_path}", flush=True)

    return X, y, g

def make_group_folds(
    y_int: torch.Tensor,
    groups: np.ndarray,
    cv_scheme: str = "loro",
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[int, np.ndarray]]:
    """
    Recording-grouped cross-validation folds.

    Every clip of a given source recording lands entirely in ONE fold's held-out
    set; no recording is shared between the training pool and the held-out fold.
    This is what makes the reported accuracy an estimate of performance on
    *unseen recordings* rather than on memorised recording conditions.

      cv_scheme="loro"        -> LeaveOneGroupOut: one fold per source recording
                                 (n_folds = number of unique groups).
      cv_scheme="group-kfold" -> StratifiedGroupKFold(n_splits): fewer folds,
                                 species proportions balanced across folds.

    Returns a list of (fold_id, test_idx) with test_idx a numpy int array.
    Deterministic for a given (groups, cv_scheme, n_splits, seed).
    """
    y_arr = y_int.numpy() if hasattr(y_int, "numpy") else np.asarray(y_int)
    groups = np.asarray(groups)
    dummy_X = np.zeros(len(y_arr))

    if cv_scheme == "loro":
        splitter = LeaveOneGroupOut()
        iterator = splitter.split(dummy_X, y_arr, groups)
    elif cv_scheme == "group-kfold":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(dummy_X, y_arr, groups)
    else:
        raise ValueError(f"Unknown cv_scheme: {cv_scheme!r} (use 'loro' or 'group-kfold')")

    folds = [(k, np.asarray(test_idx, dtype=int)) for k, (_, test_idx) in enumerate(iterator)]

    # Safety: every sample held out exactly once, no recording spans a fold boundary.
    covered = np.concatenate([ti for _, ti in folds])
    assert len(covered) == len(y_arr) and len(np.unique(covered)) == len(y_arr), \
        "fold test sets do not partition the dataset"
    for k, ti in folds:
        held = set(groups[ti].tolist())
        rest = set(groups[np.setdiff1d(np.arange(len(y_arr)), ti)].tolist())
        assert held.isdisjoint(rest), f"fold {k}: a recording appears in both held-out and training pool"

    return folds


def grouped_train_val_split(
    train_pool_idx: np.ndarray,
    groups: np.ndarray,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a fold's training pool into train / early-stopping-val by *recording*,
    so the val set is also leakage-safe (its recordings appear in neither the
    train set nor the held-out fold).

    Returns (train_idx, val_idx) as absolute indices into the full dataset.
    """
    train_pool_idx = np.asarray(train_pool_idx, dtype=int)
    g_pool = np.asarray(groups)[train_pool_idx]

    if len(np.unique(g_pool)) < 2:
        # Degenerate (only one recording in the pool for some class): no val carve.
        return train_pool_idx, train_pool_idx[:0]

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_rel, va_rel = next(gss.split(np.zeros(len(train_pool_idx)), groups=g_pool))
    return train_pool_idx[tr_rel], train_pool_idx[va_rel]

class MelDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        sr: int,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        f_min: float = 0.0,
        f_max: float = None,
        train: bool = False,
        max_time_shift_frac: float = 0.10,
        freq_mask_param: int = 12,
        time_mask_param: int = 24,
    ):

        # self.X = X
        # self.y = y

        self.X = X
        self.y = y
        self.train = train
        self.max_time_shift_frac = max_time_shift_frac

        # SpecAugment (train-only)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

        if f_max is None:
            f_max = sr / 2.0

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=2.0,            # power spectrogram (needed for AmplitudeToDB)
            normalized=False,     # leave normalization to us
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        wav = self.X[idx].unsqueeze(0)   # [1, T]
        lab = self.y[idx]

        # Train-only random time shift (invariance to call position inside the window)
        if self.train:
            T = wav.shape[1]
            max_shift = int(self.max_time_shift_frac * T)
            if max_shift > 0:
                shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
                wav = torch.roll(wav, shifts=shift, dims=1)

        mel = self.mel(wav)             # [1, n_mels, time]
        mel = self.to_db(mel)           # log-mel

        # Per-example normalization in feature space (helps a lot)
        mean = mel.mean()
        std = mel.std().clamp_min(1e-6)
        mel = (mel - mean) / std

        # Train-only SpecAugment (reduces class confusions, improves robustness)
        if self.train:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        return mel, lab



class WST1Dataset(Dataset):
    """
    Returns WST order-1 "images" for CNNs.
    We compute WST coefficients first, then slice order-1 paths and treat them like a 2D map.

    Output shape:
      [1, n_paths_order1, time_frames]
    """

    def __init__(self, SX1: torch.Tensor, y: torch.Tensor):
        self.SX1 = SX1
        self.y = y

    def __len__(self):
        return len(self.SX1)

    def __getitem__(self, idx):
        feat = self.SX1[idx].unsqueeze(0)   # [1, C1, time]
        feat = torch.log1p(feat)              # optional but often helps
        feat = (feat - feat.mean()) / (feat.std().clamp_min(1e-6))
        lab = self.y[idx]
        return feat, lab


def compute_wst_order1(
    X: torch.Tensor,
    J: int,
    Q: int,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Compute Wavelet Scattering Transform order-1 coefficients for all waveforms.

    Input:
      X [N, T] on CPU
    Output:
      SX1 [N, C1, time] on CPU

    Notes:
    - This is the expensive part; we do it once per run and cache per run if desired.
    - Requires kymatio installed.
    """
    try:
        from kymatio.torch import Scattering1D
    except Exception as e:
        raise RuntimeError(
        ) from e

    N, T = X.shape
    scattering = Scattering1D(J=J, shape=T, Q=Q).to(device)

    # Determine which channels correspond to order 1
    meta = scattering.meta()
    order1_idx = np.where(meta["order"] == 1)[0]

    SX1_list = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = X[start:start + batch_size].to(device)  # [B, T]
            SX = scattering(xb)                          # [B, C, time]
            SX1 = SX[:, order1_idx, :]                   # [B, C1, time]
            SX1_list.append(SX1.cpu())

    SX1_all = torch.cat(SX1_list, dim=0)  # [N, C1, time]
    return SX1_all


# -----------------------------
# 4) Models
# -----------------------------
class BasicBlock(nn.Module):
    """Small residual block."""
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNetSmall(nn.Module):
    """Your small ResNet-like backbone."""
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.in_ch = 16
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(self.in_ch, out_ch, stride=stride, downsample=downsample)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class TinyCNN(nn.Module):
    """
    A very small CNN baseline:
    - 3 conv blocks + global pooling
    Useful as a 'minimum viable model' baseline.
    """
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)
    
class ResizeWrapper(nn.Module):
    """
    Wrap a torchvision image model so it can handle small spectrogram inputs.

    Your mel/WST "images" can be ~64 x ~16 (very small). EfficientNet/MobileNet
    downsample a lot, so we upsample first to a stable size (e.g., 224x224).
    """
    def __init__(self, backbone: nn.Module, size: int = 224, mode: str = "bilinear"):
        super().__init__()
        self.backbone = backbone
        self.size = size
        self.mode = mode

    def forward(self, x):
        # x: [B, C, H, W]
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode=self.mode, align_corners=False)
        return self.backbone(x)


# def _replace_first_conv(model: nn.Module, in_channels: int = 1):
#     """
#     Replace the very first Conv2d layer in torchvision EfficientNet/MobileNet-like models
#     to accept in_channels instead of 3.
#     """
#     # EfficientNet / EfficientNetV2 in torchvision: model.features[0][0] is Conv2d
#     first = model.features[0][0]
#     model.features[0][0] = nn.Conv2d(
#         in_channels=in_channels,
#         out_channels=first.out_channels,
#         kernel_size=first.kernel_size,
#         stride=first.stride,
#         padding=first.padding,
#         bias=False,
#     )
#     return model

def _replace_first_conv(model: nn.Module, in_channels: int = 1):
    """
    Replace the very first Conv2d layer in torchvision EfficientNet/MobileNet-like models
    to accept in_channels instead of 3.

    If the model is pretrained on RGB and we switch to 1-channel, initialize the new
    conv by averaging RGB weights (good transfer for spectrograms).
    """
    old = model.features[0][0]
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    # Weight transfer if old conv is RGB and new is mono
    with torch.no_grad():
        if hasattr(old, "weight") and old.weight is not None and old.weight.shape[1] == 3 and in_channels == 1:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")

    model.features[0][0] = new
    return model


def build_efficientnet(model_name: str, num_classes: int, in_channels: int = 1, resize_to: int = 224) -> nn.Module:
    """
    Supports:
      - efficientnet_b0 ... efficientnet_b7
      - efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
    """
    try:
        from torchvision import models
    except Exception as e:
        raise RuntimeError("torchvision is required for EfficientNet models.") from e

    name = model_name.lower()

    # --- EfficientNet B0..B7 ---
    if name.startswith("efficientnet_b"):
        ctor = getattr(models, name, None)
        if ctor is None:
            raise ValueError(f"torchvision.models has no constructor named '{name}'.")
        
        # model = ctor(weights=None)

        # Load pretrained weights if available
        weights_enum = getattr(models, f"{name.upper()}_Weights", None)
        w = weights_enum.DEFAULT if weights_enum is not None else None
        model = ctor(weights=w)

        model = _replace_first_conv(model, in_channels=in_channels)

        # classifier is Sequential(Dropout, Linear)
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)

        return ResizeWrapper(model, size=resize_to)

    # --- EfficientNetV2 (S/M/L) ---
    if name in {"efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"}:
        ctor = getattr(models, name, None)
        if ctor is None:
            raise ValueError(
                f"torchvision.models has no constructor named '{name}'. "
                "Your torchvision may be too old for EfficientNetV2."
            )
        # model = ctor(weights=None)

        # Load pretrained weights if available
        weights_enum = getattr(models, f"{name.upper()}_Weights", None)
        w = weights_enum.DEFAULT if weights_enum is not None else None
        model = ctor(weights=w)

        model = _replace_first_conv(model, in_channels=in_channels)

        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, num_classes)

        return ResizeWrapper(model, size=resize_to)

    raise ValueError(f"Unknown EfficientNet family model: {model_name}")


# def build_mobilenetv3_small(num_classes: int, in_channels: int = 1) -> nn.Module:
#     """
#     MobileNetV3-small from torchvision, adapted for 1-channel inputs.

#     Why it's included:
#     - lightweight architecture family very different from ResNet
#     - good test of 'do we need residual nets at all?'
#     """
#     from torchvision.models import mobilenet_v3_small

#     model = mobilenet_v3_small(weights=None)

#     # Replace first conv to accept 1-channel input instead of 3-channel RGB
#     first_conv = model.features[0][0]
#     model.features[0][0] = nn.Conv2d(
#         in_channels,
#         first_conv.out_channels,
#         kernel_size=first_conv.kernel_size,
#         stride=first_conv.stride,
#         padding=first_conv.padding,
#         bias=False,
#     )

#     # Replace classifier output layer
#     in_feats = model.classifier[-1].in_features
#     model.classifier[-1] = nn.Linear(in_feats, num_classes)

#     return model

def build_mobilenetv3_small(num_classes: int, in_channels: int = 1, resize_to: int = 224) -> nn.Module:
    """
    MobileNetV3-small adapted for 1-channel inputs + resized inputs to avoid
    collapsing tiny spectrograms through heavy downsampling.
    """
    from torchvision.models import mobilenet_v3_small

        # Load pretrained weights if available
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = mobilenet_v3_small(weights=None)

    # Replace first conv to accept 1-channel input instead of 3-channel RGB
    old = model.features[0][0]
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False,
    )

    with torch.no_grad():
        if hasattr(old, "weight") and old.weight is not None and old.weight.shape[1] == 3 and in_channels == 1:
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        else:
            nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")

    model.features[0][0] = new


    # Replace classifier output layer
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)

    return ResizeWrapper(model, size=resize_to)


def build_model(model_name: str, num_classes: int, in_channels: int = 1) -> nn.Module:
    """Factory function to build a model by name."""
    name = model_name.lower()

    if name == "resnet_small":
        return ResNetSmall(num_classes=num_classes, in_channels=in_channels)
    if name == "tinycnn":
        return TinyCNN(num_classes=num_classes, in_channels=in_channels)

    # Keep MobileNet but improved (resizing)
    if name == "mobilenetv3_small":
        return build_mobilenetv3_small(num_classes=num_classes, in_channels=in_channels, resize_to=224)

    # EfficientNet family (B0..B7) + EfficientNetV2 (S/M/L)
    if name.startswith("efficientnet_b") or name.startswith("efficientnet_v2_"):
        return build_efficientnet(name, num_classes=num_classes, in_channels=in_channels, resize_to=224)

    raise ValueError(f"Unknown model_name: {model_name}")


# def build_model(model_name: str, num_classes: int, in_channels: int = 1) -> nn.Module:
#     """Factory function to build a model by name."""
#     if model_name == "resnet_small":
#         return ResNetSmall(num_classes=num_classes, in_channels=in_channels)
#     if model_name == "tinycnn":
#         return TinyCNN(num_classes=num_classes, in_channels=in_channels)
#     if model_name == "mobilenetv3_small":
#         return build_mobilenetv3_small(num_classes=num_classes, in_channels=in_channels)
#     raise ValueError(f"Unknown model_name: {model_name}")


# -----------------------------
# 5) Training + evaluation
# -----------------------------

def predict_proba(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: [N] int labels
      proba:  [N, C] float probabilities (softmax)
    """
    model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)                 # [B, C]
            pb = torch.softmax(logits, dim=1)  # [B, C]
            ys.append(yb.numpy())
            probs.append(pb.cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)

def evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a loader and return:
      y_true (numpy), y_pred (numpy)
    """
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            pred = logits.argmax(dim=1).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)

def ensemble_predict_proba(models: List[nn.Module], loader, device, weights=None):
    """
    Soft-voting ensemble: average (or weighted average) of probs.

    weights: list of floats length M, or None for uniform.
    """
    M = len(models)
    if weights is None:
        weights = [1.0 / M] * M
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    y_true = None
    P = None

    for i, model in enumerate(models):
        yt, proba = predict_proba(model, loader, device)
        if y_true is None:
            y_true = yt
            P = w[i] * proba
        else:
            # sanity check: same sample order/labels
            assert np.array_equal(y_true, yt), "Mismatch in loader order between models"
            P += w[i] * proba

    return y_true, P


def _sync_if_cuda(device: torch.device):
    """Ensure CUDA timings include all queued kernels."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_single_model_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run inference once and measure total wall-clock time.

    Returns:
      y_true, y_pred, elapsed_sec
    """
    model.eval()
    y_true, y_pred = [], []

    _sync_if_cuda(device)
    start = time.perf_counter()

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            pred = logits.argmax(dim=1).cpu().numpy()

            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())

    _sync_if_cuda(device)
    elapsed_sec = time.perf_counter() - start
    return np.array(y_true), np.array(y_pred), float(elapsed_sec)


def measure_ensemble_inference(
    models: List[nn.Module],
    loader: DataLoader,
    device: torch.device,
    weights=None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run ensemble inference once and measure total wall-clock time.

    Returns:
      y_true, proba, elapsed_sec
    """
    _sync_if_cuda(device)
    start = time.perf_counter()
    y_true, proba = ensemble_predict_proba(models, loader, device, weights=weights)
    _sync_if_cuda(device)
    elapsed_sec = time.perf_counter() - start
    return y_true, proba, float(elapsed_sec)


def count_model_parameters(model: nn.Module) -> int:
    """Count all learnable and frozen parameters in the model."""
    return int(sum(p.numel() for p in model.parameters()))


def _json_default(o):
    """Make numpy scalars / arrays JSON-serialisable (records built from
    pandas carry np.int64 etc., which json.dump rejects)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_records_table(records: List[Dict], csv_path: str, json_path: str):
    """Write both CSV and JSON summaries for a list of records."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=_json_default)

    if not records:
        pd.DataFrame().to_csv(csv_path, index=False)
        return

    rows = []
    for rec in records:
        row = {}
        for key, value in rec.items():
            if isinstance(value, (list, dict)):
                row[key] = json.dumps(value, default=_json_default)
            else:
                row[key] = value
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)


def save_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_json_path: str,
    out_csv_path: str,
) -> Dict:
    """
    Save per-class metrics plus aggregate metrics for one evaluation split.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for class_name in class_names:
        row = report[class_name]
        per_class_rows.append({
            "class_name": class_name,
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        })

    payload = {
        "per_class": per_class_rows,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "support": int(sum(row["support"] for row in per_class_rows)),
    }

    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(per_class_rows).to_csv(out_csv_path, index=False)
    return payload


def build_confusion_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Build a confusion matrix payload for embedding in summary records."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "labels": class_names,
        "matrix": cm.tolist(),
    }


def save_confusion_figure(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str,
):
    """
    Save a confusion matrix figure with class labels and per-cell counts.
    """
    plt.figure(figsize=(max(6, len(class_names) * 0.75), max(5, len(class_names) * 0.6)))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def collect_run_records(out_root: str) -> List[Dict]:
    """Collect per-run metrics from all completed run directories."""
    records = []
    for d in sorted(os.listdir(out_root)):
        run_dir = os.path.join(out_root, d)
        if not os.path.isdir(run_dir):
            continue

        candidate_paths = [
            os.path.join(run_dir, _lp("metrics.json")),
            os.path.join(run_dir, _lp("metrics_ensemble.json")),
        ]
        metrics_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if metrics_path is None:
            continue

        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        records.append(metrics)

    return records


def collect_ensemble_records(out_root: str) -> List[Dict]:
    """Collect per-ensemble records saved under out_root/ensemble_runs."""
    ens_dir = os.path.join(out_root, "ensemble_runs")
    if not os.path.exists(ens_dir):
        return []

    records = []
    for root, _, files in os.walk(ens_dir):
        for fname in sorted(files):
            if fname != _lp("ensemble_metrics.json"):
                continue
            with open(os.path.join(root, fname), "r") as f:
                records.append(json.load(f))
    return records


def build_seed_averaged_run_records(run_records: List[Dict]) -> List[Dict]:
    """
    Aggregate single-run records across seeds for the same experiment setup.
    """
    if not run_records:
        return []

    df = pd.DataFrame(run_records)
    group_cols = ["feature_type", "input_length", "model", "epochs_planned"]
    value_cols = [
        "epochs_ran",
        "best_val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_weighted_f1",
        "test_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "runtime_sec",
        "params",
        "inference_time_sec",
        "inference_time_ms_per_sample",
    ]

    records = []
    grouped = df.groupby(group_cols, dropna=False, sort=True)
    for keys, g in grouped:
        row = {col: g.iloc[0][col] for col in group_cols}
        row["seeds"] = sorted(int(s) for s in g["seed"].tolist())
        row["n_seeds"] = int(len(g))
        row["run_ids"] = g["run_id"].tolist()

        for col in value_cols:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_std"] = float(vals.std(ddof=0))

        records.append(row)

    return records


def make_ensemble_dirname(run_ids: List[str], weights=None) -> str:
    """
    Build a readable folder name from ensemble member run IDs.
    """
    safe_members = []
    for rid in run_ids:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in rid)
        safe_members.append(safe)

    name = "__PLUS__".join(safe_members)
    if weights is not None:
        weight_tag = "_".join(f"{float(w):g}" for w in weights)
        name = f"{name}__W__{weight_tag}"
    return name[:180]


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    out_dir: str,
    class_names: List[str],
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.03,
) -> Dict:
    """
    Train a model with early stopping on validation loss.
    Saves best checkpoint to out_dir/best.pt
    Saves learning curves to out_dir/curves.png
    Returns metrics dictionary.
    """
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    cw = None
    if class_weights is not None:
        cw = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        model.train()

        # ----- Train epoch -----
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss += loss.item() * yb.size(0)
            tr_correct += (logits.argmax(dim=1) == yb).sum().item()
            tr_total += yb.size(0)

        train_loss = tr_loss / tr_total
        train_acc = 100.0 * tr_correct / tr_total

        # ----- Validation epoch -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device, dtype=torch.long)
                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss += loss.item() * yb.size(0)
                val_correct += (logits.argmax(dim=1) == yb).sum().item()
                val_total += yb.size(0)

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)

        print(
            f"[Train] epoch {epoch:02d}/{max_epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}%",
            flush=True
        )

        # ----- Early stopping / checkpointing -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[Train] Early stop (no val loss improvement for {patience} epochs).", flush=True)
            break

        scheduler.step()


        # Save raw learning curves (easy to inspect after Slurm finishes)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    # Also save CSV (handy for quick plotting elsewhere)
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    print(f"[Save] Wrote history.json and history.csv to: {out_dir}", flush=True)
    
    # Plot curves

    plt.figure()
    plt.plot(hist["train_loss"], label="train loss")
    plt.plot(hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(hist["train_acc"], label="train acc")
    plt.plot(hist["val_acc"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curve.png"), dpi=200)
    plt.close()

    # Load best checkpoint for final evaluation
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))

    y_true, y_pred = evaluate(model, val_loader, device)

    metrics = {
        "val_accuracy": float(accuracy_score(y_true, y_pred)),
        "val_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "val_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "n_val": int(len(y_true)),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(hist["val_loss"])),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    save_confusion_figure(cm, class_names, os.path.join(out_dir, "confusion.png"), "Validation confusion matrix")

    val_confusion = build_confusion_payload(y_true, y_pred, class_names)

    # Save text report
    report = classification_report(y_true, y_pred, digits=4)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    val_per_class = save_per_class_metrics(
        y_true,
        y_pred,
        class_names,
        os.path.join(out_dir, "per_class_metrics_val.json"),
        os.path.join(out_dir, "per_class_metrics_val.csv"),
    )
    metrics["val_per_class_metrics_path"] = "per_class_metrics_val.json"
    metrics["val_confusion_matrix"] = val_confusion["matrix"]
    metrics["val_accuracy"] = float(val_per_class["accuracy"])
    metrics["val_macro_f1"] = float(val_per_class["macro_f1"])
    metrics["val_weighted_f1"] = float(val_per_class["weighted_f1"])

    # Save learning history
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    return metrics


# -----------------------------
# 6) Experiment runner
# -----------------------------
@dataclass
class ExpCfg:
    feature: str            # "mel" or "wst1"
    model: str              # "resnet_small" | "tinycnn" | "mobilenetv3_small"
    epochs: int             # max epochs
    seed: int               # random seed
    lr: float = 1e-3
    weight_decay: float = 1e-3
    patience: int = 7
    wst_J: int = 8          # sensible default from your trials
    wst_Q: int = 14         # sensible default from your trials


def _per_recording_table(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> List[Dict]:
    """One row per source recording: species, #clips, held-out accuracy."""
    groups = np.asarray(groups)
    rows = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        rows.append({
            "recording": g,
            "species": g.split("/")[0],
            "n_clips": int(m.sum()),
            "accuracy": float((y_pred[m] == y_true[m]).mean()),
        })
    return rows


def run_experiment(
    cfg: ExpCfg,
    X: torch.Tensor,
    y_str: np.ndarray,
    y_int: torch.Tensor,
    class_names: List[str],
    folds: List[Tuple[int, np.ndarray]],
    groups: np.ndarray,
    sr: int,
    input_length: int,
    out_root: str,
    device: torch.device,
    sx1_cache: Dict,
    keep_fold_ckpt: bool = True,
    val_frac: float = 0.15,
    cv_scheme: str = "loro",
):
    """
    Run one experiment configuration under recording-grouped cross-validation.

    For every fold: train a fresh model on the training-pool recordings (minus a
    grouped early-stopping val split), then predict on the held-out recording(s).
    Predictions from all folds are stitched into one out-of-fold (OOF) prediction
    per clip and scored once. No recording is ever seen in training and test.

    Per-fold artefacts land in <out_dir>/folds/fold_XXX/ (full train_one_run
    output, original names). Run-level outputs are all loro_-prefixed:
        loro_metrics.json, loro_classification_report_test.txt,
        loro_per_class_metrics_test.*, loro_confusion_test.png,
        loro_oof_predictions.npz, loro_per_recording_accuracy.csv,
        loro_folds_summary.csv, loro_config.json
    """
    run_id = _lp(f"{cfg.feature}__{cfg.model}__ep{cfg.epochs}__seed{cfg.seed}")
    out_dir = os.path.join(out_root, run_id)
    folds_dir = os.path.join(out_dir, "folds")
    os.makedirs(folds_dir, exist_ok=True)

    num_classes = int(y_int.max().item() + 1)
    N = len(y_int)
    groups = np.asarray(groups)
    y_true_all = y_int.numpy()
    all_idx = np.arange(N)

    print(f"\n[Run] Starting {run_id}  ({len(folds)} folds, cv={cv_scheme})", flush=True)

    # Config for reproducibility
    with open(os.path.join(out_dir, _lp("config.json")), "w") as f:
        json.dump({
            **cfg.__dict__,
            "input_length": input_length,
            "sr": sr,
            "cv_scheme": cv_scheme,
            "n_folds": len(folds),
            "val_frac": val_frac,
        }, f, indent=2, default=_json_default)

    # Order-1 scattering: compute once per (J, Q), reuse across folds AND configs.
    SX1 = None
    if cfg.feature == "wst1":
        key = (int(cfg.wst_J), int(cfg.wst_Q))
        if key not in sx1_cache:
            print(f"[WST] computing order-1 scattering for J={key[0]} Q={key[1]} (once)", flush=True)
            sx1_cache[key] = compute_wst_order1(X=X, J=key[0], Q=key[1], device=device, batch_size=512)
        SX1 = sx1_cache[key]
        with open(os.path.join(out_dir, _lp("wst_shape.txt")), "w") as f:
            f.write(f"SX1 shape: {tuple(SX1.shape)}\nJ={cfg.wst_J}, Q={cfg.wst_Q}\n")

    in_channels = 1
    batch_size = 64

    oof_pred = np.full(N, -1, dtype=int)
    oof_prob = np.zeros((N, num_classes), dtype=np.float64)
    oof_filled = np.zeros(N, dtype=bool)
    fold_rows: List[Dict] = []
    params_seen = None
    start = time.time()

    for fold_id, test_idx in folds:
        fdir = os.path.join(folds_dir, f"fold_{fold_id:03d}")
        os.makedirs(fdir, exist_ok=True)
        pred_path = os.path.join(fdir, _lp("predictions.npz"))

        # ---- resume: reuse a completed fold (retrain if the file is unreadable) ----
        if os.path.exists(pred_path):
            try:
                d = np.load(pred_path, allow_pickle=True)
                ti = d["test_idx"].astype(int)
                oof_pred[ti] = d["y_pred"]
                oof_prob[ti] = d["y_prob"]
                oof_filled[ti] = True
                fr = {"fold_id": int(fold_id), "n_test": int(len(ti)),
                      "n_recordings": int(len(set(groups[ti].tolist()))),
                      "accuracy": float((d["y_pred"] == d["y_true"]).mean())}
                for k_ in ("val_macro_f1", "epochs_ran", "best_val_loss"):
                    if k_ in d.files:
                        fr[k_] = float(d[k_])
                fold_rows.append(fr)
                print(f"[Resume] {run_id} fold {fold_id} ({fr['accuracy']:.4f})", flush=True)
                continue
            except Exception as e:
                print(f"[Resume] {run_id} fold {fold_id}: unreadable predictions.npz "
                      f"({e}); retraining this fold", flush=True)
                try:
                    os.remove(pred_path)
                except OSError:
                    pass

        seed_f = int(cfg.seed) * 1000 + int(fold_id)
        set_seed(seed_f)

        train_pool = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        tr_idx, va_idx = grouped_train_val_split(train_pool, groups, val_frac=val_frac, seed=seed_f)
        if len(va_idx) == 0:
            va_idx = tr_idx  # impossible with this data; guards train_one_run

        if cfg.feature == "mel":
            train_ds = MelDataset(X[tr_idx], y_int[tr_idx], sr=sr, n_mels=64, train=True)
            val_ds = MelDataset(X[va_idx], y_int[va_idx], sr=sr, n_mels=64, train=False)
            test_ds = MelDataset(X[test_idx], y_int[test_idx], sr=sr, n_mels=64, train=False)
        else:
            train_ds = WST1Dataset(SX1[tr_idx], y_int[tr_idx])
            val_ds = WST1Dataset(SX1[va_idx], y_int[va_idx])
            test_ds = WST1Dataset(SX1[test_idx], y_int[test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = build_model(cfg.model, num_classes=num_classes, in_channels=in_channels)
        params_seen = count_model_parameters(model)

        fold_val_metrics = train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            patience=cfg.patience,
            out_dir=fdir,
            class_names=class_names,
        )
        # train_one_run has reloaded best.pt into `model`
        y_true_f, prob_f = predict_proba(model, test_loader, device)
        y_pred_f = prob_f.argmax(axis=1)

        np.savez_compressed(
            pred_path,
            test_idx=test_idx.astype(int),
            y_true=y_true_f.astype(int),
            y_pred=y_pred_f.astype(int),
            y_prob=prob_f.astype(np.float32),
            groups=groups[test_idx],
            val_macro_f1=np.float64(fold_val_metrics.get("val_macro_f1", np.nan)),
            epochs_ran=np.float64(fold_val_metrics.get("epochs_ran", np.nan)),
            best_val_loss=np.float64(fold_val_metrics.get("best_val_loss", np.nan)),
        )
        if not keep_fold_ckpt:
            try:
                os.remove(os.path.join(fdir, "best.pt"))
            except OSError:
                pass

        oof_pred[test_idx] = y_pred_f
        oof_prob[test_idx] = prob_f
        oof_filled[test_idx] = True
        fold_rows.append({
            "fold_id": int(fold_id),
            "n_test": int(len(test_idx)),
            "n_recordings": int(len(set(groups[test_idx].tolist()))),
            "accuracy": float(accuracy_score(y_true_f, y_pred_f)),
            "val_macro_f1": float(fold_val_metrics.get("val_macro_f1", np.nan)),
            "epochs_ran": float(fold_val_metrics.get("epochs_ran", np.nan)),
            "best_val_loss": float(fold_val_metrics.get("best_val_loss", np.nan)),
        })
        print(f"[Fold] {run_id} {fold_id + 1}/{len(folds)} | "
              f"acc={fold_rows[-1]['accuracy']:.4f}", flush=True)

    assert oof_filled.all(), f"{run_id}: {(~oof_filled).sum()} samples never held out"

    if params_seen is None:  # every fold was resumed from disk
        params_seen = count_model_parameters(
            build_model(cfg.model, num_classes=num_classes, in_channels=in_channels)
        )

    # ---------- pooled out-of-fold scoring ----------
    save_confusion_figure(
        confusion_matrix(y_true_all, oof_pred), class_names,
        os.path.join(out_dir, _lp("confusion_test.png")), f"{run_id} OOF confusion matrix",
    )
    with open(os.path.join(out_dir, _lp("classification_report_test.txt")), "w") as f:
        f.write(classification_report(y_true_all, oof_pred, target_names=class_names,
                                      digits=4, zero_division=0))
    oof_per_class = save_per_class_metrics(
        y_true_all, oof_pred, class_names,
        os.path.join(out_dir, _lp("per_class_metrics_test.json")),
        os.path.join(out_dir, _lp("per_class_metrics_test.csv")),
    )

    np.savez_compressed(
        os.path.join(out_dir, _lp("oof_predictions.npz")),
        index=all_idx, y_true=y_true_all.astype(int), y_pred=oof_pred.astype(int),
        y_prob=oof_prob.astype(np.float32), groups=groups,
    )

    rec_rows = _per_recording_table(y_true_all, oof_pred, groups)
    pd.DataFrame(rec_rows).to_csv(os.path.join(out_dir, _lp("per_recording_accuracy.csv")), index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(out_dir, _lp("folds_summary.csv")), index=False)

    def _fold_mean(col):
        return float(pd.to_numeric(fold_df[col], errors="coerce").mean()) if col in fold_df else float("nan")

    val_macro_mean = _fold_mean("val_macro_f1")

    metrics = {
        "run_id": run_id,
        "feature_type": cfg.feature,
        "input_length": int(input_length),
        "model": cfg.model,
        "seed": int(cfg.seed),
        "epochs_planned": int(cfg.epochs),
        "params": int(params_seen) if params_seen is not None else None,
        "num_classes": int(num_classes),
        "classes": class_names,
        "cv_scheme": cv_scheme,
        "n_folds": int(len(folds)),
        "n_recordings": int(len(np.unique(groups))),
        # OOF == the reported test performance under grouped CV
        "test_accuracy": float(oof_per_class["accuracy"]),
        "test_macro_f1": float(oof_per_class["macro_f1"]),
        "test_weighted_f1": float(oof_per_class["weighted_f1"]),
        "n_test": int(N),
        "oof_accuracy": float(oof_per_class["accuracy"]),
        "oof_macro_f1": float(oof_per_class["macro_f1"]),
        "oof_weighted_f1": float(oof_per_class["weighted_f1"]),
        # per-recording spread (each recording weighted equally)
        "per_recording_accuracy_mean": float(np.mean([r["accuracy"] for r in rec_rows])),
        "per_recording_accuracy_std": float(np.std([r["accuracy"] for r in rec_rows])),
        # mean of the per-fold early-stopping val macro-F1 (kept so the champion
        # selector and seed-averaging code still have a val_* column)
        "val_accuracy": val_macro_mean,
        "val_macro_f1": val_macro_mean,
        "val_weighted_f1": val_macro_mean,
        "epochs_ran": _fold_mean("epochs_ran"),
        "best_val_loss": _fold_mean("best_val_loss"),
        "runtime_sec": float(time.time() - start),
        "test_per_class_metrics_path": _lp("per_class_metrics_test.json"),
        "test_confusion_matrix": confusion_matrix(y_true_all, oof_pred).tolist(),
    }
    with open(os.path.join(out_dir, _lp("metrics.json")), "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    print(
        f"[Run] Finished {run_id} | OOF acc={metrics['oof_accuracy']:.4f} "
        f"macroF1={metrics['oof_macro_f1']:.4f} | "
        f"per-recording acc {metrics['per_recording_accuracy_mean']:.4f}"
        f" +/- {metrics['per_recording_accuracy_std']:.4f}",
        flush=True,
    )


def _print_fold_composition(folds, groups, y_int, class_names):
    groups = np.asarray(groups)
    y = y_int.numpy()
    # recordings per class
    per_class_rec = {c: len(set(groups[y == i].tolist())) for i, c in enumerate(class_names)}
    print("[CV] recordings per class: " +
          ", ".join(f"{c}={n}" for c, n in per_class_rec.items()), flush=True)
    for k, ti in folds[:min(len(folds), 8)]:
        sp = sorted(set(g.split("/")[0] for g in groups[ti].tolist()))
        print(f"[CV] fold {k:>3}: {len(ti):>4} clips | "
              f"{len(set(groups[ti].tolist()))} recording(s) | {', '.join(sp)}", flush=True)
    if len(folds) > 8:
        print(f"[CV] ... ({len(folds)} folds total)", flush=True)


def _run_ensemble_oof(args, class_names):
    """Soft-voting ensemble over the members' out-of-fold predictions."""
    run_ids = [_lp(s.strip()) for s in args.ensemble_runs.split(",") if s.strip()]
    weights = None
    if args.ensemble_weights.strip():
        weights = [float(x) for x in args.ensemble_weights.split(",")]
        assert len(weights) == len(run_ids), "ensemble_weights length must match ensemble_runs"

    members = []
    for rid in run_ids:
        p = os.path.join(args.out_root, rid, _lp("oof_predictions.npz"))
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found - run the single-model grouped CV for '{rid}' first "
                "(the ensemble reuses each member's out-of-fold predictions)."
            )
        members.append(np.load(p, allow_pickle=True))

    y_true = members[0]["y_true"].astype(int)
    grp = np.asarray(members[0]["groups"])
    for rid, d in zip(run_ids[1:], members[1:]):
        if not np.array_equal(d["y_true"].astype(int), y_true):
            raise RuntimeError(
                f"OOF label order of '{rid}' differs from '{run_ids[0]}'. "
                "All members must share the same folds.json (same cv/seed/recording_id_len)."
            )

    cfgs = []
    for rid in run_ids:
        with open(os.path.join(args.out_root, rid, _lp("config.json")), "r") as f:
            cfgs.append(json.load(f))

    M = len(members)
    w = np.ones(M) / M if weights is None else np.asarray(weights, dtype=float) / float(np.sum(weights))
    P = sum(wi * d["y_prob"].astype(np.float64) for wi, d in zip(w, members))
    y_pred = P.argmax(axis=1)

    ens_root = os.path.join(args.out_root, "ensemble_runs")
    os.makedirs(ens_root, exist_ok=True)
    ensemble_dir = os.path.join(ens_root, _lp(make_ensemble_dirname(run_ids, weights=weights)))
    os.makedirs(ensemble_dir, exist_ok=True)

    save_confusion_figure(
        confusion_matrix(y_true, y_pred), class_names,
        os.path.join(ensemble_dir, _lp("confusion_test.png")), "Ensemble OOF confusion matrix",
    )
    with open(os.path.join(ensemble_dir, _lp("classification_report_test.txt")), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0))
    ens_per_class = save_per_class_metrics(
        y_true, y_pred, class_names,
        os.path.join(ensemble_dir, _lp("per_class_metrics_test.json")),
        os.path.join(ensemble_dir, _lp("per_class_metrics_test.csv")),
    )
    rec_rows = _per_recording_table(y_true, y_pred, grp)
    pd.DataFrame(rec_rows).to_csv(os.path.join(ensemble_dir, _lp("per_recording_accuracy.csv")), index=False)
    np.savez_compressed(
        os.path.join(ensemble_dir, _lp("oof_predictions.npz")),
        y_true=y_true, y_pred=y_pred.astype(int), y_prob=P.astype(np.float32), groups=grp,
    )

    ensemble_id = hashlib.sha1(("|".join(run_ids) + "|" + ",".join(map(str, weights or []))).encode()).hexdigest()[:12]
    ens_metrics = {
        "ensemble_id": ensemble_id,
        "ensemble_members": run_ids,
        "weights": weights,
        "feature_types": [c["feature"] for c in cfgs],
        "input_lengths": [int(c.get("input_length", args.T)) for c in cfgs],
        "ensemble_size": int(len(run_ids)),
        "cv_scheme": cfgs[0].get("cv_scheme", "loro"),
        "test_accuracy": float(ens_per_class["accuracy"]),
        "test_macro_f1": float(ens_per_class["macro_f1"]),
        "test_weighted_f1": float(ens_per_class["weighted_f1"]),
        "oof_accuracy": float(ens_per_class["accuracy"]),
        "oof_macro_f1": float(ens_per_class["macro_f1"]),
        "oof_weighted_f1": float(ens_per_class["weighted_f1"]),
        "n_test": int(len(y_true)),
        "per_recording_accuracy_mean": float(np.mean([r["accuracy"] for r in rec_rows])),
        "per_recording_accuracy_std": float(np.std([r["accuracy"] for r in rec_rows])),
        "ensemble_dir": os.path.relpath(ensemble_dir, args.out_root),
        "test_per_class_metrics_path": os.path.join(os.path.relpath(ensemble_dir, args.out_root), _lp("per_class_metrics_test.json")),
        "test_confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    with open(os.path.join(ensemble_dir, _lp("ensemble_metrics.json")), "w") as f:
        json.dump(ens_metrics, f, indent=2, default=_json_default)
    pd.DataFrame([{
        k: (json.dumps(v, default=_json_default) if isinstance(v, (list, dict)) else v)
        for k, v in ens_metrics.items()
    }]).to_csv(os.path.join(ensemble_dir, _lp("ensemble_metrics.csv")), index=False)

    save_records_table(
        collect_ensemble_records(args.out_root),
        os.path.join(args.out_root, _lp("ensemble_summary.csv")),
        os.path.join(args.out_root, _lp("ensemble_summary.json")),
    )
    print(f"[Ensemble] {ensemble_dir}", flush=True)
    print(f"[Ensemble] OOF acc={ens_metrics['oof_accuracy']:.4f} "
          f"macroF1={ens_metrics['oof_macro_f1']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--out_root", type=str, default="runs_loro", help="Output folder for all runs")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Where to store preprocessing cache")
    parser.add_argument("--sr", type=int, default=47600, help="Target sampling rate")
    parser.add_argument("--T", type=int, default=8000, help="Fixed waveform length (samples)")
    parser.add_argument("--min_per_class", type=int, default=50, help="Min samples per class")
    parser.add_argument("--cv", type=str, default="loro", choices=["loro", "group-kfold"],
                        help="Recording-grouped CV scheme: 'loro' = leave-one-recording-out "
                             "(one fold per recording); 'group-kfold' = StratifiedGroupKFold.")
    parser.add_argument("--n_splits", type=int, default=5, help="Folds for --cv group-kfold (ignored for loro)")
    parser.add_argument("--val_frac", type=float, default=0.15,
                        help="Fraction of each fold's training recordings held out for early-stopping val")
    parser.add_argument("--recording_id_len", type=int, default=5,
                        help="Chars of the filename stem that identify a source recording (Watkins: 5)")
    parser.add_argument("--keep_fold_checkpoints", action="store_true",
                        help="Keep every fold's best.pt (default: delete after predictions are saved)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Build the grouped cache, write folds.json, print the fold "
                             "composition, then exit before any training.")
    parser.add_argument("--ensemble_runs", type=str, default="", help="Comma-separated run_ids to soft-vote over their OOF predictions")
    parser.add_argument("--ensemble_weights", type=str, default="", help="Optional comma-separated weights, same length as ensemble_runs")
    args = parser.parse_args()

    print(f"[Paths] out_root = {os.path.abspath(args.out_root)}", flush=True)
    print(f"[Paths] cache_dir = {os.path.abspath(args.cache_dir)}", flush=True)

    device = get_device()
    print(f"[System] Device: {device}", flush=True)
    os.makedirs(args.out_root, exist_ok=True)

    # Preprocess and cache waveforms once for all runs. The "_grouped" suffix keeps
    # this cache separate from the pre-LORO one (which has no groups array).
    cache_path = os.path.join(
        args.cache_dir,
        _lp(f"preproc_sr{args.sr}_T{args.T}_min{args.min_per_class}_grouped.pt"),
    )
    X, y_str, groups = load_and_preprocess_dataset(
        dataset_root=args.data_root,
        target_sr=args.sr,
        target_len=args.T,
        min_per_class=args.min_per_class,
        cache_path=cache_path,
        recording_id_len=args.recording_id_len,
    )

    # Encode labels ONCE (fixed mapping)
    le = LabelEncoder()
    y_int = torch.tensor(le.fit_transform(y_str), dtype=torch.long)
    class_names = le.classes_.tolist()

    if args.ensemble_runs.strip():
        _run_ensemble_oof(args, class_names)
        return

    # Recording-grouped CV folds (deterministic for a given cv/n_splits/seed).
    folds = make_group_folds(y_int, groups, cv_scheme=args.cv, n_splits=args.n_splits, seed=42)
    folds_manifest = {
        "cv_scheme": args.cv,
        "n_folds": len(folds),
        "recording_id_len": args.recording_id_len,
        "val_frac": args.val_frac,
        "n_recordings": int(len(np.unique(groups))),
        "folds": {int(k): sorted(set(np.asarray(groups)[ti].tolist())) for k, ti in folds},
    }
    with open(os.path.join(args.out_root, _lp("folds.json")), "w") as f:
        json.dump(folds_manifest, f, indent=2, default=_json_default)
    print(f"[CV] scheme={args.cv} | {len(folds)} folds | "
          f"{folds_manifest['n_recordings']} recordings | saved {_lp('folds.json')}", flush=True)
    _print_fold_composition(folds, groups, y_int, class_names)

    if args.dry_run:
        print("[Dry-run] grouped cache + folds.json written, fold composition printed. "
              "No training. Remove --dry_run to run the grid.", flush=True)
        return

    # -----------------------------
    # Experiment grid (unchanged)
    # -----------------------------
    features = ["mel", "wst1"]
    models = [
        "resnet_small",
        "tinycnn",
        "mobilenetv3_small",
        "efficientnet_b0",
    ]
    epoch_budgets = [20, 40, 60]
    seeds = [0, 1, 2]

    experiments: List[ExpCfg] = []
    for feat in features:
        for model in models:
            for ep in epoch_budgets:
                for sd in seeds:
                    experiments.append(ExpCfg(feature=feat, model=model, epochs=ep, seed=sd))

    sx1_cache: Dict = {}  # order-1 scattering reused across folds AND configs
    for cfg in experiments:
        run_experiment(
            cfg,
            X,
            y_str,
            y_int,
            class_names,
            folds,
            groups,
            sr=args.sr,
            input_length=args.T,
            out_root=args.out_root,
            device=device,
            sx1_cache=sx1_cache,
            keep_fold_ckpt=args.keep_fold_checkpoints,
            val_frac=args.val_frac,
            cv_scheme=args.cv,
        )

    # After all runs, summarize.
    run_records = collect_run_records(args.out_root)
    save_records_table(
        run_records,
        os.path.join(args.out_root, _lp("summary_runs.csv")),
        os.path.join(args.out_root, _lp("summary_runs.json")),
    )
    print(f"[Done] Wrote run summaries to {os.path.join(args.out_root, _lp('summary_runs.csv'))}", flush=True)

    seed_avg_records = build_seed_averaged_run_records(run_records)
    save_records_table(
        seed_avg_records,
        os.path.join(args.out_root, _lp("summary_runs_across_seeds.csv")),
        os.path.join(args.out_root, _lp("summary_runs_across_seeds.json")),
    )
    print(f"[Done] Wrote across-seed summaries to "
          f"{os.path.join(args.out_root, _lp('summary_runs_across_seeds.csv'))}", flush=True)

    ensemble_records = collect_ensemble_records(args.out_root)
    save_records_table(
        ensemble_records,
        os.path.join(args.out_root, _lp("ensemble_summary.csv")),
        os.path.join(args.out_root, _lp("ensemble_summary.json")),
    )

    # -----------------------------
    # Champion = best config by pooled out-of-fold macro-F1 (the CV result itself
    # is the evaluation; there is no separate held-out test set).
    # -----------------------------
    df = pd.DataFrame(run_records)
    if len(df) > 0 and "oof_macro_f1" in df.columns:
        best_row = df.sort_values("oof_macro_f1", ascending=False).iloc[0]
        champion = {
            "best_run_id": best_row["run_id"],
            "cv_scheme": args.cv,
            "oof_accuracy": float(best_row["oof_accuracy"]),
            "oof_macro_f1": float(best_row["oof_macro_f1"]),
            "oof_weighted_f1": float(best_row["oof_weighted_f1"]),
            "per_recording_accuracy_mean": float(best_row.get("per_recording_accuracy_mean", float("nan"))),
            "per_recording_accuracy_std": float(best_row.get("per_recording_accuracy_std", float("nan"))),
            "n_test": int(best_row["n_test"]),
        }
        with open(os.path.join(args.out_root, _lp("final_oof_metrics.json")), "w") as f:
            json.dump(champion, f, indent=2, default=_json_default)
        print(f"[Champion] {champion['best_run_id']} | "
              f"OOF acc={champion['oof_accuracy']:.4f} macroF1={champion['oof_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
