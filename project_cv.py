"""MNIST CNN comparison for DLBAIPCV01 Project: Computer Vision.

Models: Chollet's published Simple MNIST ConvNet and a modern Keras approximation
of the LeCun et al. (1989) convolutional architecture.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
SEED = 2026
EPOCHS = 15
BATCH_SIZE = 128


def load_mnist():
    """Load the canonical 60k/10k MNIST split and apply shared preprocessing."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)


def build_chollet_model():
    """Chollet (2021) Simple MNIST ConvNet, reproduced from the published example."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="digit_image"),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_1"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_1"),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv_2"),
            layers.MaxPooling2D(pool_size=(2, 2), name="max_pool_2"),
            layers.Flatten(name="flatten"),
            layers.Dropout(0.5, seed=SEED, name="dropout"),
            layers.Dense(10, activation="softmax", name="class_probabilities"),
        ],
        name="chollet_simple_mnist_convnet",
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def build_lecun_approximation():
    """LeCun-inspired feature extraction and subsampling, implemented in Keras.

    H1: four 5x5 shared-weight feature maps. H2: 2x2 local averaging/subsampling.
    H3: twelve 5x5 feature maps. Dropout is an explicit modern regularization
    proxy for the historical H2-to-H3 sparse connectivity; it is not an exact
    structural reproduction of the original connection table. H4: subsampling.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="digit_image"),
            layers.Conv2D(4, kernel_size=(5, 5), activation="tanh", name="H1_features"),
            layers.AveragePooling2D(pool_size=(2, 2), name="H2_subsampling"),
            layers.Conv2D(12, kernel_size=(5, 5), activation="tanh", name="H3_features"),
            layers.Dropout(0.25, seed=SEED, name="H2_H3_connectivity_proxy"),
            layers.AveragePooling2D(pool_size=(2, 2), name="H4_subsampling"),
            layers.Flatten(name="flatten"),
            layers.Dense(10, activation="softmax", name="class_probabilities"),
        ],
        name="lecun_1989_modern_approximation",
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


def _run_one(name, model, x_train, y_train_oh, x_test, y_test_oh, y_test):
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train_oh,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )
    training_seconds = time.perf_counter() - start
    test_loss, test_accuracy = model.evaluate(x_test, y_test_oh, verbose=0)
    probabilities = model.predict(x_test, batch_size=512, verbose=0)
    predictions = probabilities.argmax(axis=1)
    cm = tf.math.confusion_matrix(y_test, predictions, num_classes=10).numpy()
    return {
        "name": name,
        "parameters": int(model.count_params()),
        "training_seconds": float(training_seconds),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "confusion_matrix": cm.tolist(),
        "predictions": predictions.tolist(),
    }


def run_experiment():
    """Train both models once and save the machine-readable measurements/plots."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_mnist()
    print(f"Dataset: train={x_train.shape}, test={x_test.shape}; seed={SEED}")
    print(f"Runtime: TensorFlow {tf.__version__}; Python {platform.python_version()}; {platform.processor() or platform.machine()}")

    models = [
        ("Chollet (2021)", build_chollet_model()),
        ("LeCun et al. (1989), modern approximation", build_lecun_approximation()),
    ]
    model_results = []
    for name, model in models:
        print(f"\nTraining {name} ({model.count_params():,} parameters)")
        model_results.append(_run_one(name, model, x_train, y_train_oh, x_test, y_test_oh, y_test))

    payload = {
        "experiment": "MNIST CNN comparison",
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_split": 0.1,
        "training_samples": int(len(x_train)),
        "validation_samples": int(len(x_train) * 0.1),
        "test_samples": int(len(x_test)),
        "tensorflow_version": tf.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "models": model_results,
    }
    result_path = RESULTS_DIR / "experiment_results.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _save_figures(payload)
    _save_model_parameter_table(models)
    print("\nFinal test results")
    for item in model_results:
        print(
            f"{item['name']}: loss={item['test_loss']:.5f}, "
            f"accuracy={item['test_accuracy']:.5f}, "
            f"fit={item['training_seconds']:.1f}s, params={item['parameters']:,}"
        )
    print(f"Saved results to {result_path}")
    return payload


def _save_figures(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, item in zip(axes, payload["models"]):
        ConfusionMatrixDisplay(
            confusion_matrix=np.asarray(item["confusion_matrix"]),
            display_labels=np.arange(10),
        ).plot(ax=ax, colorbar=False, values_format="d", cmap="Blues")
        ax.set_title(item["name"], fontsize=10)
    fig.suptitle("MNIST test-set confusion matrices (n = 10,000)", fontsize=14, weight="bold")
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = ["#176B87", "#D97706"]
    for ax, metric, label in zip(axes, ["accuracy", "loss"], ["Accuracy", "Loss"]):
        for color, item in zip(colors, payload["models"]):
            key = f"val_{metric}"
            ax.plot(range(1, len(item["history"][metric]) + 1), item["history"][key],
                    marker="o", markersize=3, linewidth=1.8, color=color, label=item["name"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"Validation {label.lower()}")
    fig.suptitle("Learning curves", fontsize=14, weight="bold")
    fig.savefig(FIGURES_DIR / "learning_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    names = ["Chollet (2021)", "LeCun-inspired"]
    acc = [m["test_accuracy"] * 100 for m in payload["models"]]
    bars = ax.bar(names, acc, color=colors, width=0.55)
    ax.set_ylim(max(90, min(acc) - 2), 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Held-out MNIST test accuracy", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, acc):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.1, f"{value:.2f}%", ha="center", fontsize=10)
    fig.savefig(FIGURES_DIR / "accuracy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_model_parameter_table(models):
    (x_train, _, _), (x_test, _, _) = load_mnist()
    rows = []
    for name, model in models:
        for layer in model.layers:
            rows.append({"model": name, "layer": layer.name, "output_shape": str(layer.output.shape), "parameters": int(layer.count_params())})
    out = RESULTS_DIR / "layer_parameters.csv"
    import csv
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "layer", "output_shape", "parameters"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_experiment()
