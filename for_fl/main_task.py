from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

try:
    from .sec_ops import (
        GradientProtectionConfig,
        OpacusProtectionConfig,
        add_clipped_gradient_sum,
        build_clipped_gradient_sum,
        enable_opacus_protection,
        normalize_privacy_backend,
        set_noisy_average_gradients,
    )
except ImportError:
    from sec_ops import (
        GradientProtectionConfig,
        OpacusProtectionConfig,
        add_clipped_gradient_sum,
        build_clipped_gradient_sum,
        enable_opacus_protection,
        normalize_privacy_backend,
        set_noisy_average_gradients,
    )


CIFAR100_NUM_CLASSES = 100
DEFAULT_LANDMARK_NUM_CLASSES = 203_094
DEFAULT_IMAGE_SIZE = 128
DEFAULT_VAL_RATIO = 0.2


@dataclass(frozen=True)
class DataSettings:
    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = CIFAR100_NUM_CLASSES + DEFAULT_LANDMARK_NUM_CLASSES
    landmark_num_classes: int = DEFAULT_LANDMARK_NUM_CLASSES
    landmark_label_offset: int = CIFAR100_NUM_CLASSES
    include_cifar100: bool = True
    include_landmarks: bool = False
    gld_root: str | None = None
    gld_train_csv: str | None = None
    gld_val_csv: str | None = None
    gld_label_map_csv: str | None = None
    gld_verify_files: bool = False
    val_ratio: float = DEFAULT_VAL_RATIO


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.silu(self.norm(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class Net(nn.Module):
    def __init__(
        self,
        num_classes: int = CIFAR100_NUM_CLASSES + DEFAULT_LANDMARK_NUM_CLASSES,
        embedding_dim: int = 512,
        width: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, width), num_channels=width),
            nn.SiLU(),
            DepthwiseSeparableConv(width, width * 2, stride=2),
            ResidualBlock(width * 2),
            DepthwiseSeparableConv(width * 2, width * 4, stride=2),
            ResidualBlock(width * 4),
            DepthwiseSeparableConv(width * 4, width * 8, stride=2),
            ResidualBlock(width * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 8, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.embedding(x)
        return self.classifier(x)


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.lower().strip() in {"1", "true", "yes", "on"}
    return bool(value)


def settings_from_config(config: dict[str, Any] | None) -> DataSettings:
    config = config or {}
    gld_root = config.get("gld-root")
    gld_train_csv = config.get("gld-train-csv")
    include_cifar100 = _config_bool(config, "include-cifar100", True)
    include_landmarks = _config_bool(
        config,
        "include-landmarks",
        bool(gld_root and gld_train_csv),
    )
    landmark_num_classes = int(
        config.get("landmark-num-classes", DEFAULT_LANDMARK_NUM_CLASSES)
    )
    landmark_label_offset = CIFAR100_NUM_CLASSES if include_cifar100 else 0
    default_num_classes = CIFAR100_NUM_CLASSES if include_cifar100 else 0
    if include_landmarks:
        default_num_classes += landmark_num_classes
    num_classes = int(config.get("num-classes", default_num_classes))
    return DataSettings(
        image_size=int(config.get("image-size", DEFAULT_IMAGE_SIZE)),
        num_classes=num_classes,
        landmark_num_classes=landmark_num_classes,
        landmark_label_offset=landmark_label_offset,
        include_cifar100=include_cifar100,
        include_landmarks=include_landmarks,
        gld_root=gld_root,
        gld_train_csv=gld_train_csv,
        gld_val_csv=config.get("gld-val-csv"),
        gld_label_map_csv=config.get("gld-label-map-csv"),
        gld_verify_files=_config_bool(config, "gld-verify-files", False),
        val_ratio=float(config.get("val-ratio", DEFAULT_VAL_RATIO)),
    )


def _image_transforms(image_size: int) -> Compose:
    return Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _stable_bucket(value: str, modulo: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def _is_validation_sample(sample_id: str, val_ratio: float) -> bool:
    bucket = _stable_bucket(sample_id, 10_000)
    return bucket < int(val_ratio * 10_000)


class Cifar100Dataset(Dataset):
    def __init__(self, split: str, image_size: int):
        from datasets import load_dataset

        self.dataset = load_dataset("uoft-cs/cifar100", split=split)
        self.transforms = _image_transforms(image_size)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        image = item["img"].convert("RGB")
        label = int(item.get("fine_label", item.get("label")))
        return {"img": self.transforms(image), "label": torch.tensor(label)}

    def sample_key(self, index: int) -> str:
        return f"cifar100:{index}"


class PartitionedDataset(Dataset):
    def __init__(self, dataset: Dataset, partition_id: int, num_partitions: int):
        self.dataset = dataset
        self.indices = [
            idx
            for idx in range(len(dataset))
            if _stable_bucket(self._sample_key(idx), num_partitions) == partition_id
        ]

    def _sample_key(self, index: int) -> str:
        sample_key = getattr(self.dataset, "sample_key", None)
        if sample_key is not None:
            return sample_key(index)
        return str(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.dataset[self.indices[index]]


def _gld_image_path(root: Path, image_id: str) -> Path:
    return root / image_id[0] / image_id[1] / image_id[2] / f"{image_id}.jpg"


def _load_landmark_label_map(
    train_csv: Path,
    explicit_label_map_csv: Path | None,
    landmark_num_classes: int,
) -> dict[int, int]:
    if explicit_label_map_csv is not None and explicit_label_map_csv.exists():
        with explicit_label_map_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            return {
                int(row["landmark_id"]): int(row["class_index"])
                for row in reader
                if row.get("landmark_id") and row.get("class_index")
            }

    counts: Counter[int] = Counter()
    with train_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("landmark_id"):
                counts[int(row["landmark_id"])] += 1

    most_common = counts.most_common(landmark_num_classes)
    return {landmark_id: idx for idx, (landmark_id, _) in enumerate(most_common)}


class GoogleLandmarksV2Dataset(Dataset):
    def __init__(
        self,
        root: str,
        metadata_csv: str,
        label_map: dict[int, int],
        image_size: int,
        split: str,
        val_ratio: float,
        label_offset: int = CIFAR100_NUM_CLASSES,
        partition_id: int | None = None,
        num_partitions: int | None = None,
        verify_files: bool = False,
    ):
        self.root = Path(root)
        self.transforms = _image_transforms(image_size)
        self.samples: list[tuple[str, int]] = []

        with Path(metadata_csv).open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                image_id = row.get("id")
                landmark_id_raw = row.get("landmark_id")
                if not image_id or not landmark_id_raw:
                    continue
                landmark_id = int(landmark_id_raw)
                if landmark_id not in label_map:
                    continue
                if partition_id is not None and num_partitions is not None:
                    if _stable_bucket(f"gld:{image_id}", num_partitions) != partition_id:
                        continue
                is_val = _is_validation_sample(f"gld:{image_id}", val_ratio)
                if split == "train" and is_val:
                    continue
                if split in {"val", "test"} and not is_val:
                    continue
                if verify_files and not _gld_image_path(self.root, image_id).exists():
                    continue
                label = label_offset + label_map[landmark_id]
                self.samples.append((image_id, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_id, label = self.samples[index]
        image = Image.open(_gld_image_path(self.root, image_id)).convert("RGB")
        return {"img": self.transforms(image), "label": torch.tensor(label)}


def _build_datasets(
    settings: DataSettings,
    partition_id: int | None = None,
    num_partitions: int | None = None,
    split: str = "train",
) -> list[Dataset]:
    datasets: list[Dataset] = []

    if settings.include_cifar100:
        cifar_split = "test" if split in {"val", "test"} else "train"
        cifar_dataset: Dataset = Cifar100Dataset(cifar_split, settings.image_size)
        if partition_id is not None and num_partitions is not None:
            cifar_dataset = PartitionedDataset(cifar_dataset, partition_id, num_partitions)
        datasets.append(cifar_dataset)

    if settings.include_landmarks:
        if not settings.gld_root or not settings.gld_train_csv:
            raise ValueError(
                "GLDv2 is enabled, but 'gld-root' and 'gld-train-csv' are not set."
            )
        train_csv = Path(settings.gld_train_csv)
        label_map_csv = (
            Path(settings.gld_label_map_csv) if settings.gld_label_map_csv else None
        )
        label_map = _load_landmark_label_map(
            train_csv,
            label_map_csv,
            settings.landmark_num_classes,
        )
        metadata_csv = settings.gld_val_csv if split in {"val", "test"} else settings.gld_train_csv
        datasets.append(
            GoogleLandmarksV2Dataset(
                root=settings.gld_root,
                metadata_csv=metadata_csv or settings.gld_train_csv,
                label_map=label_map,
                image_size=settings.image_size,
                split=split,
                val_ratio=settings.val_ratio,
                label_offset=settings.landmark_label_offset,
                partition_id=partition_id,
                num_partitions=num_partitions,
                verify_files=settings.gld_verify_files,
            )
        )

    if not datasets:
        raise ValueError("No datasets enabled. Enable CIFAR-100 and/or GLDv2.")
    return datasets


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    settings: DataSettings | None = None,
):
    settings = settings or DataSettings()
    train_dataset = ConcatDataset(
        _build_datasets(settings, partition_id, num_partitions, split="train")
    )
    val_dataset = ConcatDataset(
        _build_datasets(settings, partition_id, num_partitions, split="val")
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    testloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    return trainloader, testloader


def load_centralized_dataset(
    batch_size: int = 128,
    settings: DataSettings | None = None,
):
    settings = settings or DataSettings()
    dataset = ConcatDataset(_build_datasets(settings, split="test"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    grad_clip_norm: float | None = None,
    grad_noise_std: float = 0.0,
    privacy_backend: str = "manual_gradient_protection",
    opacus_noise_multiplier: float | None = None,
    opacus_accountant: str = "prv",
    opacus_delta: float = 1e-5,
    opacus_secure_mode: bool = False,
    opacus_poisson_sampling: bool = True,
    opacus_grad_sample_mode: str = "hooks",
) -> float:
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    privacy_backend = normalize_privacy_backend(privacy_backend)
    gradient_protection = GradientProtectionConfig(
        clip_norm=grad_clip_norm,
        noise_std=grad_noise_std,
    )
    opacus_state = None
    if privacy_backend == "opacus":
        opacus_state = enable_opacus_protection(
            net,
            optimizer,
            trainloader,
            OpacusProtectionConfig(
                noise_multiplier=max(
                    grad_noise_std if opacus_noise_multiplier is None else opacus_noise_multiplier,
                    0.0,
                ),
                max_grad_norm=grad_clip_norm or 1.0,
                accountant=opacus_accountant,
                secure_mode=opacus_secure_mode,
                poisson_sampling=opacus_poisson_sampling,
                grad_sample_mode=opacus_grad_sample_mode,
                delta=opacus_delta,
            ),
        )
        net = opacus_state.model
        optimizer = opacus_state.optimizer
        trainloader = opacus_state.trainloader

    net.train()
    running_loss = 0.0
    try:
        for _ in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                if privacy_backend == "manual_gradient_protection":
                    running_loss += _train_manual_private_batch(
                        net,
                        images,
                        labels,
                        criterion,
                        optimizer,
                        gradient_protection,
                    )
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
    finally:
        if opacus_state is not None:
            epsilon = opacus_state.get_epsilon(opacus_delta)
            if epsilon is not None:
                print(f"Opacus privacy spent: epsilon={epsilon:.4f}, delta={opacus_delta}")
            opacus_state.cleanup()
    return running_loss / max(1, epochs * len(trainloader))


def _train_manual_private_batch(
    net: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_protection: GradientProtectionConfig,
) -> float:
    parameters = list(net.parameters())
    accumulator: list[torch.Tensor | None] = [None for _ in parameters]
    batch_loss = 0.0

    for sample_index in range(images.shape[0]):
        optimizer.zero_grad(set_to_none=True)
        sample_images = images[sample_index : sample_index + 1]
        sample_labels = labels[sample_index : sample_index + 1]
        sample_loss = criterion(net(sample_images), sample_labels)
        sample_loss.backward()
        clipped_grads, _ = build_clipped_gradient_sum(net, gradient_protection)
        add_clipped_gradient_sum(accumulator, clipped_grads)
        batch_loss += float(sample_loss.detach().item())

    optimizer.zero_grad(set_to_none=True)
    set_noisy_average_gradients(
        net,
        accumulator,
        sample_count=int(images.shape[0]),
        config=gradient_protection,
    )
    optimizer.step()
    return batch_loss / max(1, int(images.shape[0]))


def _classification_f1(
    true_positive: torch.Tensor,
    predicted_count: torch.Tensor,
    target_count: torch.Tensor,
) -> tuple[float, float]:
    precision = true_positive / predicted_count.clamp_min(1)
    recall = true_positive / target_count.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    observed = target_count > 0
    if observed.any():
        macro_f1 = float(f1[observed].mean().item())
        weighted_f1 = float(
            (f1[observed] * target_count[observed]).sum().item()
            / target_count[observed].sum().clamp_min(1).item()
        )
    else:
        macro_f1 = 0.0
        weighted_f1 = 0.0
    return macro_f1, weighted_f1


def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    num_classes = int(getattr(net, "num_classes", 0))
    true_positive = torch.zeros(num_classes, dtype=torch.float64)
    predicted_count = torch.zeros(num_classes, dtype=torch.float64)
    target_count = torch.zeros(num_classes, dtype=torch.float64)
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            predictions = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item()
            correct += (predictions == labels).sum().item()

            batch_targets = labels.detach().cpu()
            batch_predictions = predictions.detach().cpu()
            if num_classes > 0:
                target_count += torch.bincount(
                    batch_targets,
                    minlength=num_classes,
                ).to(dtype=torch.float64)
                predicted_count += torch.bincount(
                    batch_predictions,
                    minlength=num_classes,
                ).to(dtype=torch.float64)
                true_positive += torch.bincount(
                    batch_targets[batch_targets == batch_predictions],
                    minlength=num_classes,
                ).to(dtype=torch.float64)
    accuracy = correct / max(1, len(testloader.dataset))
    loss = loss / max(1, len(testloader))
    macro_f1, weighted_f1 = _classification_f1(
        true_positive,
        predicted_count,
        target_count,
    )
    return loss, accuracy, macro_f1, weighted_f1
