#!/usr/bin/env python3
"""Train a handwritten digit CNN and export model + inference config.

Default dataset: MNIST (0-9 handwritten digits).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CNN for handwritten digit recognition (MNIST)."
    )
    parser.add_argument(
        "--model-out",
        default="models/udise_digit_model.keras",
        help="Output model path (.keras recommended)",
    )
    parser.add_argument(
        "--config-out",
        default=None,
        help="Output JSON config path. Default: <model-out>.json",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Validation split from training data",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=28,
        help="Input image size (MNIST native is 28)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable light data augmentation during training",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_mnist(image_size: int):
    import tensorflow as tf

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if image_size != 28:
        x_train = tf.image.resize(x_train[..., None], (image_size, image_size)).numpy()[..., 0]
        x_test = tf.image.resize(x_test[..., None], (image_size, image_size)).numpy()[..., 0]

    # MNIST digits are white on black background; keep as-is and
    # use --invert in inference preprocessing for alignment.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def build_model(image_size: int, learning_rate: float, augment: bool):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 1), name="digit")
    x = inputs

    if augment:
        x = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.03),
                tf.keras.layers.RandomZoom(0.08),
                tf.keras.layers.RandomTranslation(0.05, 0.05),
            ],
            name="augment",
        )(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax", name="digit_class")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_digit_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.image_size <= 0:
        raise RuntimeError("--image-size must be positive")
    if not (0.0 < args.validation_split < 1.0):
        raise RuntimeError("--validation-split must be between 0 and 1")

    # TensorFlow import here so users who only infer with torch are not blocked.
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)

    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist(args.image_size)

    print("Building model...")
    model = build_model(
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        augment=args.augment,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    print("Training...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating on test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)

    config_out = Path(args.config_out) if args.config_out else Path(str(model_out) + ".json")
    config = {
        "framework": "keras",
        "image_size": args.image_size,
        # Keep True because MNIST is white-digit-on-black. This aligns with pipeline preprocess.
        "invert": True,
        "digit_count": 11,
        "dataset": "mnist",
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "epochs_ran": int(len(history.history.get("loss", []))),
    }
    config_out.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Saved model: {model_out}")
    print(f"Saved config: {config_out}")
    print("Use this model directly with udise_ocr_pipeline.py (it auto-loads .json config).")


if __name__ == "__main__":
    main()
