"""Flower / PyTorch task code for CIFAR-100 + Google Landmarks v2."""

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


CIFAR100_NUM_CLASSES = 100
DEFAULT_LANDMARK_NUM_CLASSES = 203_094
DEFAULT_IMAGE_SIZE = 128
DEFAULT_VAL_RATIO = 0.2


@dataclass(frozen=True)
class DataSettings:
    """Shared settings used by the server and every federated client."""

    image_size: int = DEFAULT_IMAGE_SIZE
    num_classes: int = CIFAR100_NUM_CLASSES + DEFAULT_LANDMARK_NUM_CLASSES
    landmark_num_classes: int = DEFAULT_LANDMARK_NUM_CLASSES
    include_cifar100: bool = True
    include_landmarks: bool = False
    gld_root: str | None = None
    gld_train_csv: str | None = None
    gld_val_csv: str | None = None
    gld_label_map_csv: str | None = None
    gld_verify_files: bool = False
    val_ratio: float = DEFAULT_VAL_RATIO


class DepthwiseSeparableConv(nn.Module):
    """A compact CNN block that scales better than plain 5x5 convolutions."""

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
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.silu(self.bn(x))


class ResidualBlock(nn.Module):
    """Residual block for large mixed image datasets."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels)
        self.conv2 = DepthwiseSeparableConv(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class Net(nn.Module):
    """Scalable CNN for the combined CIFAR-100 + GLDv2 classification task."""

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
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
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
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.embedding(x)
        return self.classifier(x)


def settings_from_config(config: dict[str, Any] | None) -> DataSettings:
    """Build data/model settings from Flower run config."""

    config = config or {}
    gld_root = config.get("gld-root")
    gld_train_csv = config.get("gld-train-csv")
    include_landmarks = bool(
        config.get("include-landmarks", bool(gld_root and gld_train_csv))
    )
    landmark_num_classes = int(
        config.get("landmark-num-classes", DEFAULT_LANDMARK_NUM_CLASSES)
    )
    default_num_classes = CIFAR100_NUM_CLASSES
    if include_landmarks:
        default_num_classes += landmark_num_classes
    num_classes = int(config.get("num-classes", default_num_classes))
    return DataSettings(
        image_size=int(config.get("image-size", DEFAULT_IMAGE_SIZE)),
        num_classes=num_classes,
        landmark_num_classes=landmark_num_classes,
        include_cifar100=bool(config.get("include-cifar100", True)),
        include_landmarks=include_landmarks,
        gld_root=gld_root,
        gld_train_csv=gld_train_csv,
        gld_val_csv=config.get("gld-val-csv"),
        gld_label_map_csv=config.get("gld-label-map-csv"),
        gld_verify_files=bool(config.get("gld-verify-files", False)),
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
    """CIFAR-100 split with labels occupying class ids 0..99."""

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
    """Hash-partition any dataset into stable federated client shards."""

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
    """Map sparse GLDv2 landmark ids into a compact class range after CIFAR-100."""

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
    """Lazy local GLDv2 dataset backed by metadata CSV and extracted JPG files."""

    def __init__(
        self,
        root: str,
        metadata_csv: str,
        label_map: dict[int, int],
        image_size: int,
        split: str,
        val_ratio: float,
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
                label = CIFAR100_NUM_CLASSES + label_map[landmark_id]
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
    """Load one federated client shard from CIFAR-100 and optional GLDv2."""

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
    )
    testloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader


def load_centralized_dataset(
    batch_size: int = 128,
    settings: DataSettings | None = None,
):
    """Load central validation data for the server-side global evaluation."""

    settings = settings or DataSettings()
    dataset = ConcatDataset(_build_datasets(settings, split="test"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    grad_clip_norm: float | None = None,
    grad_noise_std: float = 0.0,
) -> float:
    """Train the model on a local federated client dataset."""

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(net(images), labels)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip_norm)

            # Подключение "шифрования" градиентов через шум:
            # перед отправкой обновлений в федеративный контур клиент может добавлять
            # гауссов шум к локальным градиентам. Это не заменяет криптографическое
            # secure aggregation, но маскирует вклад отдельного примера/клиента и
            # обычно используется как DP-SGD-подобный слой приватности.
            if grad_noise_std > 0:
                for parameter in net.parameters():
                    if parameter.grad is not None:
                        noise = torch.normal(
                            mean=0.0,
                            std=grad_noise_std,
                            size=parameter.grad.shape,
                            device=parameter.grad.device,
                        )
                        parameter.grad.add_(noise)

            optimizer.step()
            running_loss += loss.item()
    return running_loss / max(1, epochs * len(trainloader))


def test(net: nn.Module, testloader: DataLoader, device: torch.device):
    """Validate the model on local or centralized validation data."""

    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / max(1, len(testloader.dataset))
    loss = loss / max(1, len(testloader))
    return loss, accuracy
