from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "datasets"
LABEL_DISPLAY_MAP = {
    0: "Foto Asli",
    1: "Gambar AI",
}

SPLIT_CONFIG: dict[str, dict[str, Any]] = {
    "train": {
        "csv_name": "train.csv",
        "id_column": "file_name",
        "label_column": "label",
        "title": "Training Set",
        "description": "Labeled images used to train the classifier.",
    },
    "test": {
        "csv_name": "test.csv",
        "id_column": "id",
        "label_column": None,
        "title": "Test Set",
        "description": "Unlabeled evaluation split referenced by the competition CSV.",
    },
    "test_v2": {
        "csv_name": "test_v2.csv",
        "labels_csv_name": "test_v2_labels.csv",
        "id_column": "id",
        "label_column": "label",
        "title": "Test V2",
        "description": "Holdout split with optional labels for validation and review.",
    },
}


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.loc[:, ~frame.columns.str.contains(r"^Unnamed")].copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def _dataset_csv(csv_name: str) -> Path:
    return DATASET_DIR / csv_name


def _resolve_image_path(relative_path: str) -> Path:
    return DATASET_DIR / Path(relative_path)


def format_label_name(value: Any) -> str:
    if pd.isna(value):
        return "Belum berlabel"

    normalized_value = int(value)
    return LABEL_DISPLAY_MAP.get(normalized_value, f"Label {normalized_value}")


@lru_cache(maxsize=None)
def _load_split_frame(split_name: str) -> pd.DataFrame:
    if split_name not in SPLIT_CONFIG:
        raise KeyError(f"Unknown split: {split_name}")

    config = SPLIT_CONFIG[split_name]
    frame = _clean_frame(pd.read_csv(_dataset_csv(config["csv_name"])))

    id_column = config["id_column"]
    if id_column not in frame.columns:
        available = ", ".join(frame.columns.astype(str))
        raise KeyError(f"Column '{id_column}' not found in {config['csv_name']}. Columns: {available}")

    prepared = frame.rename(columns={id_column: "relative_path"}).copy()

    labels_csv_name = config.get("labels_csv_name")
    label_column = config.get("label_column")

    if labels_csv_name:
        labels = _clean_frame(pd.read_csv(_dataset_csv(labels_csv_name))).rename(columns={id_column: "relative_path"})
        if label_column and label_column in labels.columns:
            prepared = prepared.merge(labels[["relative_path", label_column]], on="relative_path", how="left")

    if label_column and label_column in prepared.columns:
        prepared["label"] = pd.to_numeric(prepared[label_column], errors="coerce")
    else:
        prepared["label"] = pd.Series([pd.NA] * len(prepared), dtype="Float64")

    prepared["absolute_path"] = prepared["relative_path"].map(_resolve_image_path)
    prepared["file_exists"] = prepared["absolute_path"].map(Path.exists)
    prepared["split"] = split_name
    prepared["split_title"] = config["title"]
    prepared["label_display"] = prepared["label"].map(format_label_name)
    return prepared


def load_split_frame(split_name: str) -> pd.DataFrame:
    return _load_split_frame(split_name).copy()


def list_available_splits() -> list[str]:
    return list(SPLIT_CONFIG)


def get_label_distribution(split_name: str) -> pd.DataFrame:
    frame = load_split_frame(split_name)
    labeled = frame.dropna(subset=["label"]).copy()
    if labeled.empty:
        return pd.DataFrame(columns=["label", "count"])

    distribution = (
        labeled.groupby("label", dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("label")
        .reset_index(drop=True)
    )
    distribution["label_name"] = distribution["label"].map(format_label_name)
    return distribution


def get_split_summary(split_name: str) -> dict[str, Any]:
    frame = load_split_frame(split_name)
    labeled = frame.dropna(subset=["label"])
    missing_count = int((~frame["file_exists"]).sum())
    present_count = int(frame["file_exists"].sum())
    class_distribution = get_label_distribution(split_name)

    return {
        "split": split_name,
        "title": SPLIT_CONFIG[split_name]["title"],
        "description": SPLIT_CONFIG[split_name]["description"],
        "rows": int(len(frame)),
        "present_files": present_count,
        "missing_files": missing_count,
        "labeled_rows": int(len(labeled)),
        "class_balance": (
            " / ".join(f"{row.label_name}: {int(row.count):,}" for row in class_distribution.itertuples())
            if not class_distribution.empty
            else "No labels"
        ),
    }


def get_overview_table() -> pd.DataFrame:
    rows = [get_split_summary(split_name) for split_name in list_available_splits()]
    return pd.DataFrame(rows)


def sample_records(
    split_name: str,
    sample_size: int = 12,
    label_filter: str = "All",
    existing_only: bool = True,
    seed: int = 7,
) -> pd.DataFrame:
    frame = load_split_frame(split_name)

    if existing_only:
        frame = frame[frame["file_exists"]]

    if label_filter != "All":
        if label_filter == "Unlabeled" or label_filter == "Belum berlabel":
            frame = frame[frame["label"].isna()]
        else:
            label_lookup = {display_name: label_value for label_value, display_name in LABEL_DISPLAY_MAP.items()}
            label_value = label_lookup.get(label_filter)
            if label_value is None:
                label_value = int(str(label_filter).replace("Class ", "").replace("Label ", ""))
            frame = frame[frame["label"] == label_value]

    if frame.empty:
        return frame

    size = min(sample_size, len(frame))
    return frame.sample(n=size, random_state=seed).reset_index(drop=True)


def get_missing_files(split_name: str, limit: int = 50) -> pd.DataFrame:
    frame = load_split_frame(split_name)
    missing = frame.loc[~frame["file_exists"], ["relative_path", "split_title"]].copy()
    return missing.head(limit)


def estimate_image_stats(split_name: str, sample_size: int = 64) -> dict[str, float | int | None]:
    frame = sample_records(split_name, sample_size=sample_size, existing_only=True)
    if frame.empty:
        return {"sample_size": 0, "avg_width": None, "avg_height": None}

    widths: list[int] = []
    heights: list[int] = []

    for image_path in frame["absolute_path"]:
        with Image.open(image_path) as image:
            widths.append(image.width)
            heights.append(image.height)

    return {
        "sample_size": len(widths),
        "avg_width": int(sum(widths) / len(widths)),
        "avg_height": int(sum(heights) / len(heights)),
    }
