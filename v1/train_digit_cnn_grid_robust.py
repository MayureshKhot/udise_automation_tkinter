#!/usr/bin/env python3
"""Train a grid-line-robust handwritten digit model.

Design goals:
- CPU-friendly (i3 class machines)
- More robust to OMR vertical separator lines
- Exports .keras model + inference config JSON
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train robust handwritten digit CNN with grid-line augmentation.")
    p.add_argument("--model-out", default="models/udise_digit_model_robust.keras", help="Output model path")
    p.add_argument("--config-out", default=None, help="Output config JSON path (default: <model-out>.json)")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--validation-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image-size", type=int, default=28)
    p.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_mnist(image_size: int):
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if image_size != 28:
        x_train = tf.image.resize(x_train[..., None], (image_size, image_size)).numpy()[..., 0]
        x_test = tf.image.resize(x_test[..., None], (image_size, image_size)).numpy()[..., 0]

    # MNIST: white digits on black background. Matches inference with invert=True.
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def _random_shift_rotate(img_u8: np.ndarray) -> np.ndarray:
    h, w = img_u8.shape[:2]
    angle = np.random.uniform(-10, 10)
    tx = np.random.uniform(-0.18 * w, 0.18 * w)
    ty = np.random.uniform(-0.12 * h, 0.12 * h)
    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
    mat[0, 2] += tx
    mat[1, 2] += ty
    return cv2.warpAffine(img_u8, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _random_thickness(img_u8: np.ndarray) -> np.ndarray:
    r = np.random.rand()
    if r < 0.33:
        k = np.ones((2, 2), np.uint8)
        return cv2.dilate(img_u8, k, iterations=1)
    if r < 0.66:
        k = np.ones((2, 2), np.uint8)
        return cv2.erode(img_u8, k, iterations=1)
    return img_u8


def _add_vertical_grid_artifacts(img_u8: np.ndarray) -> np.ndarray:
    h, w = img_u8.shape[:2]
    out = img_u8.copy()

    # Simulate OMR separator lines near edges (common failure source).
    if np.random.rand() < 0.8:
        t = np.random.randint(1, 3)
        x = np.random.randint(0, min(5, w))
        cv2.line(out, (x, 0), (x, h - 1), 255, thickness=t)

    if np.random.rand() < 0.8:
        t = np.random.randint(1, 3)
        x = np.random.randint(max(0, w - 5), w)
        cv2.line(out, (x, 0), (x, h - 1), 255, thickness=t)

    # Occasionally add an internal thin line.
    if np.random.rand() < 0.2:
        t = 1
        x = np.random.randint(max(1, w // 4), min(w - 1, (3 * w) // 4))
        cv2.line(out, (x, 0), (x, h - 1), 255, thickness=t)

    return out


def _add_noise(img_u8: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 6.0, size=img_u8.shape).astype(np.float32)
    out = img_u8.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def augment_digit_np(img_f32: np.ndarray) -> np.ndarray:
    """Input [H,W,1] float32 [0,1] -> output same format."""
    img = (img_f32[..., 0] * 255.0).astype(np.uint8)
    img = _random_shift_rotate(img)
    img = _random_thickness(img)
    img = _add_vertical_grid_artifacts(img)
    img = _add_noise(img)
    out = (img.astype(np.float32) / 255.0)[..., None]
    return out


def build_dataset(x: np.ndarray, y: np.ndarray, image_size: int, batch_size: int, training: bool, augment: bool):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(20000, len(x)), reshuffle_each_iteration=True)

    if training and augment:
        def _map_aug(img, label):
            img2 = tf.numpy_function(augment_digit_np, [img], tf.float32)
            img2.set_shape((image_size, image_size, 1))
            return img2, label
        ds = ds.map(_map_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(learning_rate: float):
    import tensorflow as tf

    # Dynamic spatial input + GAP makes model tolerant to minor input-size variation.
    inp = tf.keras.Input(shape=(None, None, 1), name="digit")

    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(48, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(96, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(10, activation="softmax", name="digit_class")(x)

    model = tf.keras.Model(inp, out, name="udise_digit_cnn_grid_robust")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(args: argparse.Namespace) -> Tuple[float, float, int]:
    import tensorflow as tf

    if args.image_size <= 0:
        raise RuntimeError("--image-size must be positive")
    if not (0.0 < args.validation_split < 1.0):
        raise RuntimeError("--validation-split must be between 0 and 1")

    tf.keras.utils.set_random_seed(args.seed)

    (x_train, y_train), (x_test, y_test) = load_mnist(args.image_size)

    # Split train/val
    n = len(x_train)
    n_val = int(round(n * args.validation_split))
    n_val = max(1, min(n - 1, n_val))

    x_val, y_val = x_train[:n_val], y_train[:n_val]
    x_tr, y_tr = x_train[n_val:], y_train[n_val:]

    ds_train = build_dataset(x_tr, y_tr, args.image_size, args.batch_size, training=True, augment=not args.no_augment)
    ds_val = build_dataset(x_val, y_val, args.image_size, args.batch_size, training=False, augment=False)
    ds_test = build_dataset(x_test, y_test, args.image_size, args.batch_size, training=False, augment=False)

    model = build_model(args.learning_rate)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6),
    ]

    hist = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=callbacks, verbose=1)

    test_loss, test_acc = model.evaluate(ds_test, verbose=0)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    cfg_path = Path(args.config_out) if args.config_out else Path(str(model_out) + ".json")
    cfg = {
        "framework": "keras",
        "image_size": args.image_size,
        "invert": True,
        "digit_count": 11,
        "dataset": "mnist+grid_artifact_augmentation",
        "model_family": "depthwise_separable_cnn_dynamic_input",
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_ran": int(len(hist.history.get("loss", []))),
        "recommended_preprocess": {
            "remove_vertical_lines": True,
            "line_min_height_ratio": 0.75,
            "line_max_width": 3,
            "line_edge_margin": 4,
        },
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return float(test_acc), float(test_loss), int(len(hist.history.get("loss", [])))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    acc, loss, epochs = train(args)
    model_out = Path(args.model_out)
    cfg_path = Path(args.config_out) if args.config_out else Path(str(model_out) + ".json")

    print(f"Saved model: {model_out}")
    print(f"Saved config: {cfg_path}")
    print(f"Epochs: {epochs}")
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
