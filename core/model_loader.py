from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "model" / "best_model.pth"
DEFAULT_IMAGE_SIZE = 224


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.se = SqueezeExcitation(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        features = self.conv(x)
        features = self.se(features)
        return self.relu(features + residual)


class DetectorBinaryClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [32, 64, 128, 256, 512, 512]
        in_channels = 3
        blocks: list[nn.Module] = []

        for out_channels in channels:
            blocks.append(ResidualSEBlock(in_channels, out_channels))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


@dataclass(frozen=True)
class PredictionResult:
    probability: float
    predicted_label: int
    raw_logit: float


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> DetectorBinaryClassifier:
    return DetectorBinaryClassifier()


def _extract_state_dict(checkpoint: Any) -> dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            nested_state_dict = checkpoint.get(key)
            if isinstance(nested_state_dict, dict):
                checkpoint = nested_state_dict
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("Unsupported checkpoint format: expected a state dict or checkpoint dictionary.")

    state_dict: dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        state_dict[key.removeprefix("module.")] = value
    return state_dict


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return get_default_device()
    return torch.device(device)


def _build_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    device: str | torch.device | None = None,
) -> DetectorBinaryClassifier:
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(Path(model_path), map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    model = build_model()
    model.load_state_dict(state_dict)
    model.to(resolved_device)
    model.eval()
    return model


@lru_cache(maxsize=2)
def load_cached_model(device: str | None = None) -> DetectorBinaryClassifier:
    return load_model(device=device)


def preprocess_image(image: Image.Image, image_size: int = DEFAULT_IMAGE_SIZE) -> Tensor:
    rgb_image = image.convert("RGB")
    tensor = _build_transform(image_size=image_size)(rgb_image)
    return tensor.unsqueeze(0)


def predict_image(
    image: Image.Image | str | Path,
    model: DetectorBinaryClassifier | None = None,
    device: str | torch.device | None = None,
) -> PredictionResult:
    resolved_device = _resolve_device(device)
    inference_model = model if model is not None else load_cached_model(str(resolved_device))

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened_image:
            batch = preprocess_image(opened_image).to(resolved_device)
    else:
        batch = preprocess_image(image).to(resolved_device)

    with torch.inference_mode():
        logits = inference_model(batch).squeeze()
        probability = torch.sigmoid(logits).item()

    return PredictionResult(
        probability=probability,
        predicted_label=int(probability >= 0.5),
        raw_logit=float(logits.item()),
    )
