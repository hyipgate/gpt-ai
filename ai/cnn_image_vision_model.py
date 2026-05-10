from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from ai.image_feedback import load_image_feedback
from ai.image_vision_model import _direction_suffix
from core.config import resolve_path


@dataclass(frozen=True)
class CnnVisionPrediction:
    probability: float
    passed: bool
    threshold: float
    model_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for the local CNN vision model. "
            "Install it with: python -m pip install tensorflow"
        ) from exc
    return tf


def image_tensor(image_path: str | Path, size: int = 96) -> np.ndarray:
    from PIL import Image, ImageOps

    path = resolve_path(image_path)
    with Image.open(path) as image:
        image = ImageOps.grayscale(image)
        image = ImageOps.fit(image, (size, size), method=Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array[..., np.newaxis]


def _build_dataset(feedback: pd.DataFrame, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    images = []
    directions = []
    labels = []
    weights = []
    kept = []
    for row in feedback.itertuples(index=False):
        image_path = resolve_path(row.image_path)
        if not image_path.exists():
            continue
        images.append(image_tensor(image_path, size=size))
        directions.append([_direction_suffix(row.direction)])
        labels.append(int(row.label))
        weights.append(float(row.weight))
        kept.append(str(image_path))
    if not images:
        return (
            np.empty((0, size, size, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            [],
        )
    return (
        np.stack(images).astype(np.float32),
        np.asarray(directions, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        kept,
    )


def build_cnn_model(size: int = 96):
    tf = _load_tensorflow()
    image_input = tf.keras.Input(shape=(size, size, 1), name="chart_image")
    direction_input = tf.keras.Input(shape=(1,), name="direction")

    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(image_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Concatenate()([x, direction_input])
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="quality_probability")(x)

    model = tf.keras.Model(inputs=[image_input, direction_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def train_cnn_image_vision_model(
    feedback_path: str | Path = "ai/models/manual_image_feedback.csv",
    model_path: str | Path = "ai/models/cnn_image_vision_model.keras",
    metadata_path: str | Path = "ai/models/cnn_image_vision_model.metadata.json",
    size: int = 96,
    threshold: float = 0.60,
    epochs: int = 30,
    batch_size: int = 16,
) -> dict[str, object]:
    tf = _load_tensorflow()
    feedback = load_image_feedback(feedback_path)
    if len(feedback) < 20:
        raise ValueError(f"Need at least 20 labeled images for CNN training; got {len(feedback)}.")
    if feedback["label"].nunique() < 2:
        raise ValueError("Image feedback contains only one class; add both good and bad examples.")

    images, directions, labels, weights, kept = _build_dataset(feedback, size)
    if len(labels) < 20:
        raise ValueError(f"Need at least 20 readable labeled images; got {len(labels)}.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Readable image labels contain only one class.")
    class_counts = pd.Series(labels).value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "CNN training needs at least 2 readable images in each class "
            f"for validation splitting; got counts={class_counts.to_dict()}."
        )

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    model = build_cnn_model(size=size)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", patience=3, factor=0.5),
    ]
    history = model.fit(
        {"chart_image": images[train_idx], "direction": directions[train_idx]},
        labels[train_idx],
        sample_weight=weights[train_idx],
        validation_data=(
            {"chart_image": images[test_idx], "direction": directions[test_idx]},
            labels[test_idx],
            weights[test_idx],
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    probabilities = model.predict({"chart_image": images[test_idx], "direction": directions[test_idx]}, verbose=0).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels[test_idx], predictions, average="binary", zero_division=0)
    metrics: dict[str, float | list[float]] = {
        "samples": float(len(labels)),
        "train_samples": float(len(train_idx)),
        "test_samples": float(len(test_idx)),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "image_size": float(size),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "epochs_ran": float(len(history.history.get("loss", []))),
    }
    if len(np.unique(labels[test_idx])) > 1:
        metrics["test_roc_auc"] = float(roc_auc_score(labels[test_idx], probabilities))

    output = resolve_path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metadata = {
        "model_path": str(output),
        "size": size,
        "threshold": threshold,
        "metrics": metrics,
        "training_images": kept,
        "objective": "local_cnn_screenshot_chart_quality",
    }
    metadata_output = resolve_path(metadata_path)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def predict_cnn_image_vision(
    image_path: str | Path | None,
    direction: str | int = "",
    model_path: str | Path = "ai/models/cnn_image_vision_model.keras",
    metadata_path: str | Path = "ai/models/cnn_image_vision_model.metadata.json",
) -> CnnVisionPrediction | None:
    if not image_path:
        return None
    model_file = resolve_path(model_path)
    metadata_file = resolve_path(metadata_path)
    if not model_file.exists() or not metadata_file.exists():
        return None
    tf = _load_tensorflow()
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    size = int(metadata.get("size", 96))
    threshold = float(metadata.get("threshold", 0.60))
    image = image_tensor(image_path, size=size)[np.newaxis, ...]
    direction_array = np.asarray([[_direction_suffix(str(direction))]], dtype=np.float32)
    model = tf.keras.models.load_model(model_file)
    probability = float(model.predict({"chart_image": image, "direction": direction_array}, verbose=0).reshape(-1)[0])
    return CnnVisionPrediction(
        probability=probability,
        passed=probability >= threshold,
        threshold=threshold,
        model_path=str(model_file),
    )
