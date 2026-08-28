import os
import json
import time
import platform
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader

import torchaudio
from sklearn.preprocessing import LabelEncoder


# =========================================================
# Utilities
# =========================================================

def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def center_crop_or_pad(x: torch.Tensor, target_len: int) -> torch.Tensor:
    T = x.shape[1]
    if T == target_len:
        return x
    if T < target_len:
        add = target_len - T
        left = add // 2
        right = add - left
        return pad(x, (left, right))
    center = T // 2
    half = target_len // 2
    start = max(0, center - half)
    end = start + target_len
    return x[:, start:end]


def standardize_per_signal(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = X.mean(dim=1, keepdim=True)
    std = X.std(dim=1, keepdim=True)
    return (X - mean) / (std + eps)


def load_and_preprocess_dataset(
    dataset_root: str,
    target_sr: int,
    target_len: int,
    min_per_class: int,
    cache_path: Optional[str] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    if cache_path is not None and os.path.exists(cache_path):
        print(f"[Cache] Loading preprocessed data from: {cache_path}")
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        return ckpt["X"], ckpt["y"]

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root does not exist or is not a directory: {dataset_root}")

    print("[Data] Loading wav files...")
    waveforms = []
    labels = []

    for root, _, files in os.walk(dataset_root):
        for fname in files:
            if not fname.lower().endswith(".wav"):
                continue

            full_path = os.path.join(root, fname)
            rel = os.path.relpath(full_path, dataset_root)
            class_name = rel.split(os.sep)[0]

            try:
                x, sr = sf.read(full_path, dtype="float32")

                if x.ndim == 1:
                    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                else:
                    x = torch.tensor(x.T, dtype=torch.float32)

                if x.shape[0] > 1:
                    x = x.mean(dim=0, keepdim=True)

                if sr != target_sr:
                    new_n = int(x.shape[1] * target_sr / sr)
                    x_rs = resample(x.numpy(), new_n, axis=1)
                    x = torch.tensor(x_rs, dtype=torch.float32)

                x = center_crop_or_pad(x, target_len)

                waveforms.append(x)
                labels.append(class_name)

            except Exception as e:
                print(f"[Warn] Skipping {full_path}: {e}")

    if not waveforms:
        zip_files = [
            fname for fname in os.listdir(dataset_root)
            if fname.lower().endswith(".zip")
        ]
        if zip_files:
            raise RuntimeError(
                "No .wav files were found under dataset_root. "
                "The directory currently contains .zip archives, so extract them into "
                "class subfolders before running the benchmark."
            )
        raise RuntimeError(f"No .wav files were found under dataset_root: {dataset_root}")

    X = torch.stack(waveforms, dim=0).squeeze(1)
    y = np.array(labels)

    # remove exact duplicates
    import pandas as pd
    df = pd.DataFrame(X.numpy())
    df["y"] = y
    df = df.drop_duplicates(subset=list(range(target_len)))
    X = torch.tensor(df.iloc[:, :target_len].values, dtype=torch.float32)
    y = df["y"].values

    keep = []
    for cname in np.unique(y):
        idx = np.where(y == cname)[0]
        if len(idx) > min_per_class:
            keep.extend(idx.tolist())
    keep = np.array(sorted(keep))

    X = X[keep]
    y = y[keep]
    X = standardize_per_signal(X)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"X": X, "y": y}, cache_path)
        print(f"[Cache] Saved preprocessed data to: {cache_path}")

    return X, y


def make_fixed_split(
    y_int: torch.Tensor,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    from sklearn.model_selection import train_test_split

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    idx_all = np.arange(len(y_int))

    idx_trval, idx_te = train_test_split(
        idx_all,
        test_size=test_frac,
        random_state=seed,
        stratify=y_int.numpy(),
    )

    val_relative = val_frac / (train_frac + val_frac)
    idx_tr, idx_va = train_test_split(
        idx_trval,
        test_size=val_relative,
        random_state=seed,
        stratify=y_int.numpy()[idx_trval],
    )

    return {"train": idx_tr, "val": idx_va, "test": idx_te}


# =========================================================
# Datasets / features
# =========================================================

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
        f_max: Optional[float] = None,
        train: bool = False,
    ):
        self.X = X
        self.y = y
        self.train = train

        if f_max is None:
            f_max = sr / 2.0

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            normalized=False,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        wav = self.X[idx].unsqueeze(0)
        lab = self.y[idx]

        mel = self.mel(wav)
        mel = self.to_db(mel)
        mean = mel.mean()
        std = mel.std().clamp_min(1e-6)
        mel = (mel - mean) / std
        return mel, lab


class WST1Dataset(Dataset):
    def __init__(self, SX1: torch.Tensor, y: torch.Tensor):
        self.SX1 = SX1
        self.y = y

    def __len__(self):
        return len(self.SX1)

    def __getitem__(self, idx):
        feat = self.SX1[idx].unsqueeze(0)
        feat = torch.log1p(feat)
        feat = (feat - feat.mean()) / (feat.std().clamp_min(1e-6))
        return feat, self.y[idx]


def compute_wst_order1(
    X: torch.Tensor,
    J: int,
    Q: int,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    from kymatio.torch import Scattering1D

    N, T = X.shape
    scattering = Scattering1D(J=J, shape=T, Q=Q).to(device)
    meta = scattering.meta()
    order1_idx = np.where(meta["order"] == 1)[0]

    chunks = []
    with torch.no_grad():
        for start in range(0, N, batch_size):
            xb = X[start:start + batch_size].to(device)
            SX = scattering(xb)
            SX1 = SX[:, order1_idx, :]
            chunks.append(SX1.cpu())

    return torch.cat(chunks, dim=0)


# =========================================================
# Models
# =========================================================

class BasicBlock(nn.Module):
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
    def __init__(self, backbone: nn.Module, size: int = 224, mode: str = "bilinear"):
        super().__init__()
        self.backbone = backbone
        self.size = size
        self.mode = mode

    def forward(self, x):
        if x.shape[-2] != self.size or x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode=self.mode, align_corners=False)
        return self.backbone(x)


def _replace_first_conv(model: nn.Module, in_channels: int = 1):
    old = model.features[0][0]
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old.out_channels,
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
    return model


def build_efficientnet(model_name: str, num_classes: int, in_channels: int = 1, resize_to: int = 224):
    from torchvision import models

    name = model_name.lower()
    ctor = getattr(models, name, None)
    if ctor is None:
        raise ValueError(f"torchvision.models has no constructor named '{name}'")

    weights_enum = getattr(models, f"{name.upper()}_Weights", None)
    w = weights_enum.DEFAULT if weights_enum is not None else None
    model = ctor(weights=w)
    model = _replace_first_conv(model, in_channels=in_channels)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return ResizeWrapper(model, size=resize_to)


def build_mobilenetv3_small(num_classes: int, in_channels: int = 1, resize_to: int = 224):
    from torchvision.models import mobilenet_v3_small
    try:
        from torchvision.models import MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = mobilenet_v3_small(weights=None)

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
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return ResizeWrapper(model, size=resize_to)


def build_model(model_name: str, num_classes: int, in_channels: int = 1):
    name = model_name.lower()
    if name == "resnet_small":
        return ResNetSmall(num_classes=num_classes, in_channels=in_channels)
    if name == "tinycnn":
        return TinyCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "mobilenetv3_small":
        return build_mobilenetv3_small(num_classes=num_classes, in_channels=in_channels, resize_to=224)
    if name.startswith("efficientnet_b"):
        return build_efficientnet(name, num_classes=num_classes, in_channels=in_channels, resize_to=224)
    raise ValueError(f"Unknown model_name: {model_name}")


# =========================================================
# Benchmark helpers
# =========================================================

def count_model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def benchmark_feature_extraction_mel(
    X_wave: torch.Tensor,
    sr: int,
    device: torch.device,
    repeats: int = 100,
    warmup: int = 20,
    n_mels: int = 64,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict:
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        normalized=False,
    ).to(device)
    to_db = torchaudio.transforms.AmplitudeToDB(stype="power").to(device)

    x = X_wave.to(device)  # [B, T]

    def _run_once(inp):
        feat = mel_tf(inp.unsqueeze(1))
        feat = to_db(feat)
        mean = feat.mean(dim=(1, 2, 3), keepdim=True)
        std = feat.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        feat = (feat - mean) / std
        return feat

    with torch.no_grad():
        for _ in range(warmup):
            _ = _run_once(x)

        times = []
        for _ in range(repeats):
            sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = _run_once(x)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    batch_size = int(x.shape[0])
    return {
        "repeats": int(repeats),
        "warmup": int(warmup),
        "batch_size": batch_size,
        "mean_batch_latency_sec": float(times.mean()),
        "std_batch_latency_sec": float(times.std(ddof=0)),
        "mean_sample_latency_sec": float(times.mean() / batch_size),
        "throughput_samples_per_sec": float(batch_size / times.mean()),
    }


def benchmark_feature_extraction_wst1(
    X_wave: torch.Tensor,
    J: int,
    Q: int,
    device: torch.device,
    repeats: int = 50,
    warmup: int = 10,
) -> Dict:
    from kymatio.torch import Scattering1D

    x = X_wave.to(device)  # [B, T]
    _, T = x.shape
    scattering = Scattering1D(J=J, shape=T, Q=Q).to(device)
    meta = scattering.meta()
    order1_idx = np.where(meta["order"] == 1)[0]

    def _run_once(inp):
        sx = scattering(inp)
        sx1 = sx[:, order1_idx, :]
        sx1 = torch.log1p(sx1)
        mean = sx1.mean(dim=(1, 2), keepdim=True)
        std = sx1.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        sx1 = (sx1 - mean) / std
        return sx1.unsqueeze(1)

    with torch.no_grad():
        for _ in range(warmup):
            _ = _run_once(x)

        times = []
        for _ in range(repeats):
            sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = _run_once(x)
            sync_if_cuda(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    batch_size = int(x.shape[0])
    return {
        "repeats": int(repeats),
        "warmup": int(warmup),
        "batch_size": batch_size,
        "mean_batch_latency_sec": float(times.mean()),
        "std_batch_latency_sec": float(times.std(ddof=0)),
        "mean_sample_latency_sec": float(times.mean() / batch_size),
        "throughput_samples_per_sec": float(batch_size / times.mean()),
    }


def build_test_loader_and_wave_batch(
    feature: str,
    X: torch.Tensor,
    y_int: torch.Tensor,
    splits: Dict[str, np.ndarray],
    sr: int,
    batch_size: int,
    device: torch.device,
    J: int = 8,
    Q: int = 14,
):
    idx_te = splits["test"]
    X_test = X[idx_te]
    y_test = y_int[idx_te]

    wave_batch = X_test[:batch_size]

    if feature == "mel":
        ds = MelDataset(X_test, y_test, sr=sr, n_mels=64, train=False)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        xb, _ = next(iter(loader))
        return loader, wave_batch, xb

    if feature == "wst1":
        SX1 = compute_wst_order1(X_test, J=J, Q=Q, device=device, batch_size=min(256, batch_size))
        ds = WST1Dataset(SX1, y_test)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        xb, _ = next(iter(loader))
        return loader, wave_batch, xb

    raise ValueError(f"Unknown feature: {feature}")


def benchmark_ensemble_forward(
    models: List[nn.Module],
    member_feat_batches: List[torch.Tensor],
    device: torch.device,
    repeats: int = 100,
    warmup: int = 20,
) -> Dict:
    """
    Times the combined soft-voting forward pass of all ensemble members on the
    same batch: every member runs a forward pass, outputs are softmax'd, and the
    per-class probabilities are averaged. Mirrors ensemble_predict_proba's math
    in whale_ensembles.py, but as a timed loop instead of a one-shot prediction.
    """
    for m in models:
        m.eval()
    xs = [fb.to(device) for fb in member_feat_batches]
    batch_size = int(xs[0].shape[0])

    def _run_once():
        probs_sum = None
        for model, x in zip(models, xs):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
        return probs_sum / len(models)

    with torch.no_grad():
        for _ in range(warmup):
            _ = _run_once()

        times = []
        for _ in range(repeats):
            sync_if_cuda(device)
            t0 = time.perf_counter()
            _ = _run_once()
            sync_if_cuda(device)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times, dtype=np.float64)
    return {
        "repeats": int(repeats),
        "warmup": int(warmup),
        "batch_size": batch_size,
        "n_members": len(models),
        "mean_batch_latency_sec": float(times.mean()),
        "std_batch_latency_sec": float(times.std(ddof=0)),
        "mean_sample_latency_sec": float(times.mean() / batch_size),
        "throughput_samples_per_sec": float(batch_size / times.mean()),
    }


def find_member_run_dir(models_roots: List[str], run_id: str) -> Optional[str]:
    for models_root in models_roots:
        run_dir = os.path.join(models_root, run_id)
        if os.path.exists(os.path.join(run_dir, "config.json")) and os.path.exists(os.path.join(run_dir, "best.pt")):
            return run_dir
    return None


def load_member_config(run_dir: str, default_sr: int) -> Dict:
    with open(os.path.join(run_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return {
        "run_id": os.path.basename(run_dir),
        "run_dir": run_dir,
        "feature": cfg.get("feature", metrics.get("feature_type")),
        "model_name": cfg.get("model", metrics.get("model")),
        "input_length": int(cfg.get("input_length", metrics.get("input_length", 32000))),
        "sr": int(cfg.get("sr", default_sr)),
        "J": int(cfg.get("wst_J", 8)),
        "Q": int(cfg.get("wst_Q", 14)),
    }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark real-time inference cost (feature extraction + soft-voting forward pass) "
                     "for saved model ensembles produced by whale_ensembles.py."
    )
    parser.add_argument("--data_root", default="./data/_common-frequent", type=str, help="Dataset root directory")
    parser.add_argument(
        "--models_root",
        type=str,
        nargs="+",
        default=["./ensemble_models/ensemble members", "./best_models"],
        help="One or more folders containing the individual run subfolders referenced by each ensemble "
             "(searched in order; first match wins).",
    )
    parser.add_argument("--ensembles_root", type=str, default="./ensemble_models/final models",
                         help="Folder containing ensemble subfolders, each with an ensemble_metrics.json")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--sr", type=int, default=47600)
    parser.add_argument("--min_per_class", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="e.g. cpu, cuda:0")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 64])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="",
        help="Suffix for output filenames, e.g. '_rpi5', to avoid overwriting existing benchmark results",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"[System] device = {device}")

    if not os.path.isdir(args.ensembles_root):
        raise FileNotFoundError(f"ensembles_root does not exist or is not a directory: {args.ensembles_root}")
    valid_models_roots = [r for r in args.models_root if os.path.isdir(r)]
    if not valid_models_roots:
        raise FileNotFoundError(f"None of the models_root directories exist: {args.models_root}")

    ensemble_dirs = [
        os.path.join(args.ensembles_root, entry)
        for entry in sorted(os.listdir(args.ensembles_root))
        if os.path.isdir(os.path.join(args.ensembles_root, entry))
    ]

    data_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, Dict[str, np.ndarray], List[str]]] = {}
    results = []

    for ensemble_dir in ensemble_dirs:
        ensemble_folder_name = os.path.basename(ensemble_dir)
        metrics_path = os.path.join(ensemble_dir, "ensemble_metrics.json")
        if not os.path.exists(metrics_path):
            print(f"[Warn] Skipping {ensemble_folder_name}: no ensemble_metrics.json found")
            continue

        with open(metrics_path, "r") as f:
            ensemble_metrics = json.load(f)

        member_run_ids = ensemble_metrics.get("ensemble_members", [])
        ensemble_id = ensemble_metrics.get("ensemble_id", ensemble_folder_name)

        member_run_dirs = {run_id: find_member_run_dir(valid_models_roots, run_id) for run_id in member_run_ids}
        missing_members = [run_id for run_id, run_dir in member_run_dirs.items() if run_dir is None]
        if missing_members:
            print(
                f"[Warn] Skipping {ensemble_folder_name}: missing checkpoint/config for member(s) "
                f"{missing_members} under any of {valid_models_roots}. "
                "Copy those run folders over to benchmark this ensemble."
            )
            continue

        member_cfgs = [load_member_config(member_run_dirs[run_id], args.sr) for run_id in member_run_ids]

        print(f"\n[Ensemble] Benchmarking {ensemble_folder_name} | members={member_run_ids}")

        for c in member_cfgs:
            data_key = (c["sr"], c["input_length"])
            if data_key not in data_cache:
                sr, input_length = data_key
                cache_path = os.path.join(
                    args.cache_dir,
                    f"preproc_sr{sr}_T{input_length}_min{args.min_per_class}.pt"
                )
                X, y_str = load_and_preprocess_dataset(
                    dataset_root=args.data_root,
                    target_sr=sr,
                    target_len=input_length,
                    min_per_class=args.min_per_class,
                    cache_path=cache_path,
                )
                le = LabelEncoder()
                y_int = torch.tensor(le.fit_transform(y_str), dtype=torch.long)
                splits = make_fixed_split(y_int, seed=42, train_frac=0.70, val_frac=0.15, test_frac=0.15)
                data_cache[data_key] = (X, y_int, splits, list(le.classes_))

        for c in member_cfgs:
            X, y_int, splits, class_names = data_cache[(c["sr"], c["input_length"])]
            model = build_model(c["model_name"], num_classes=len(class_names), in_channels=1).to(device)
            model.load_state_dict(torch.load(os.path.join(c["run_dir"], "best.pt"), map_location=device))
            model.eval()
            c["model_obj"] = model
            c["params"] = count_model_parameters(model)

        total_params = int(sum(c["params"] for c in member_cfgs))

        ensemble_result = {
            "ensemble_id": ensemble_id,
            "ensemble_folder": ensemble_folder_name,
            "members": member_run_ids,
            "n_members": len(member_cfgs),
            "member_features": [c["feature"] for c in member_cfgs],
            "member_models": [c["model_name"] for c in member_cfgs],
            "member_input_lengths": [c["input_length"] for c in member_cfgs],
            "device": str(device),
            "torch_version": torch.__version__,
            "num_threads": int(torch.get_num_threads()),
            "platform": platform.platform(),
            "total_params": total_params,
            "batch_results": [],
        }

        for batch_size in args.batch_sizes:
            feat_group_cache = {}
            member_feat_batches = []

            for c in member_cfgs:
                group_key = (c["feature"], c["sr"], c["input_length"], c["J"], c["Q"])
                if group_key not in feat_group_cache:
                    X, y_int, splits, _ = data_cache[(c["sr"], c["input_length"])]
                    _, wave_batch, feat_batch = build_test_loader_and_wave_batch(
                        feature=c["feature"],
                        X=X,
                        y_int=y_int,
                        splits=splits,
                        sr=c["sr"],
                        batch_size=batch_size,
                        device=device,
                        J=c["J"],
                        Q=c["Q"],
                    )
                    if c["feature"] == "mel":
                        feat_stats = benchmark_feature_extraction_mel(
                            X_wave=wave_batch, sr=c["sr"], device=device,
                            repeats=args.repeats, warmup=args.warmup,
                        )
                    else:
                        feat_stats = benchmark_feature_extraction_wst1(
                            X_wave=wave_batch, J=c["J"], Q=c["Q"], device=device,
                            repeats=max(20, args.repeats // 2), warmup=max(5, args.warmup // 2),
                        )
                    feat_group_cache[group_key] = (feat_batch, feat_stats)

                feat_batch, _ = feat_group_cache[group_key]
                member_feat_batches.append(feat_batch)

            total_feat_time = float(sum(fs["mean_batch_latency_sec"] for _, fs in feat_group_cache.values()))

            models_list = [c["model_obj"] for c in member_cfgs]
            forward_stats = benchmark_ensemble_forward(
                models=models_list,
                member_feat_batches=member_feat_batches,
                device=device,
                repeats=args.repeats,
                warmup=args.warmup,
            )

            end_to_end_mean = total_feat_time + forward_stats["mean_batch_latency_sec"]
            end_to_end_sample = end_to_end_mean / batch_size
            signal_duration_sec = member_cfgs[0]["input_length"] / member_cfgs[0]["sr"]
            ft_model = forward_stats["mean_sample_latency_sec"] / signal_duration_sec
            ft_end_to_end = end_to_end_sample / signal_duration_sec

            batch_record = {
                "batch_size": batch_size,
                "signal_duration_sec": float(signal_duration_sec),
                "n_unique_feature_groups": len(feat_group_cache),
                "feature_extraction_total": {
                    "mean_batch_latency_sec": total_feat_time,
                },
                "ensemble_forward": forward_stats,
                "realtime_factors": {
                    "Ft_model": float(ft_model),
                    "Ft_end_to_end": float(ft_end_to_end),
                },
                "end_to_end": {
                    "mean_batch_latency_sec": float(end_to_end_mean),
                    "mean_sample_latency_sec": float(end_to_end_sample),
                    "throughput_samples_per_sec": float(batch_size / end_to_end_mean),
                },
            }
            ensemble_result["batch_results"].append(batch_record)

        out_path = os.path.join(ensemble_dir, f"inference_benchmark_ensemble{args.result_suffix}.json")
        with open(out_path, "w") as f:
            json.dump(ensemble_result, f, indent=2)
        print(f"[Save] {out_path}")

        results.append(ensemble_result)

    summary_rows = []
    for rec in results:
        for br in rec["batch_results"]:
            summary_rows.append({
                "ensemble_id": rec["ensemble_id"],
                "ensemble_folder": rec["ensemble_folder"],
                "n_members": rec["n_members"],
                "member_features": ",".join(rec["member_features"]),
                "member_models": ",".join(rec["member_models"]),
                "device": rec["device"],
                "total_params": rec["total_params"],
                "batch_size": br["batch_size"],
                "signal_duration_sec": br["signal_duration_sec"],
                "feature_time_ms_total": 1000.0 * br["feature_extraction_total"]["mean_batch_latency_sec"] / br["batch_size"],
                "forward_time_ms_per_sample": 1000.0 * br["ensemble_forward"]["mean_sample_latency_sec"],
                "Ft_model": br["realtime_factors"]["Ft_model"],
                "end_to_end_time_ms_per_sample": 1000.0 * br["end_to_end"]["mean_sample_latency_sec"],
                "Ft_end_to_end": br["realtime_factors"]["Ft_end_to_end"],
                "end_to_end_throughput_samples_per_sec": br["end_to_end"]["throughput_samples_per_sec"],
            })

    if summary_rows:
        import pandas as pd
        df = pd.DataFrame(summary_rows)
        summary_csv = os.path.join(args.ensembles_root, f"inference_benchmark_ensemble_summary{args.result_suffix}.csv")
        df.to_csv(summary_csv, index=False)
        print(f"[Done] Wrote summary to {summary_csv}")


if __name__ == "__main__":
    main()