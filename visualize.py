#!/usr/bin/env python3
"""Build the data behind the Digit Reader's 3D visualizer (visualize.html).

It loads your trained Keras model and, if you have one, the training-history JSON,
evaluates the model on the MNIST test set, and writes everything the visualizer
page needs into one file: exported_model/viz_data.json

    python visualize.py                              # auto-detect the model and history
    python visualize.py --history history.json       # point at your training history
    python visualize.py --mnist-npz mnist.npz        # use a local copy of the MNIST test set
    python visualize.py --no-eval                    # skip the test-set evaluation

Save the history from your training notebook like this:

    import json
    json.dump(history.history, open("history.json", "w"))

Then start the API (uvicorn main:app) and open /visualize. The server re-reads
viz_data.json on every request, so you can re-run this script without restarting.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = APP_DIR / "exported_model" / "mnist_cnn.keras"
DEFAULT_OUT = APP_DIR / "exported_model" / "viz_data.json"
HISTORY_CANDIDATES = (
    "exported_model/history.json",
    "exported_model/training_history.json",
    "history.json",
    "training_history.json",
    "training_log.json",
)
KEY_ALIASES = {"acc": "accuracy", "val_acc": "val_accuracy"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _keras() -> Any:
    try:
        import keras

        return keras
    except ImportError:
        from tensorflow import keras

        return keras


def load_model(path: Path) -> Any:
    if not path.is_file():
        sys.exit(f"Model not found: {path}\nUse --model to point at your .keras file.")
    return _keras().models.load_model(path)


def _normalize_history(raw: Any) -> dict[str, list[float]]:
    """Accept history.history, {'history': {...}}, or a list of per-epoch dicts."""
    if isinstance(raw, dict) and isinstance(raw.get("history"), dict):
        raw = raw["history"]
    if isinstance(raw, list):  # [{"epoch": 0, "loss": ..}, ...]
        keys = {k for row in raw if isinstance(row, dict) for k in row if k != "epoch"}
        raw = {k: [row.get(k) for row in raw] for k in keys}
    if not isinstance(raw, dict):
        raise ValueError("Unrecognised history format")
    series: dict[str, list[float]] = {}
    for key, values in raw.items():
        if not isinstance(values, list) or not values:
            continue
        try:
            series[KEY_ALIASES.get(key, key)] = [float(v) for v in values]
        except (TypeError, ValueError):
            continue
    return series


def load_history(explicit: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [explicit] if explicit else [APP_DIR / name for name in HISTORY_CANDIDATES]
    for path in candidates:
        if path and path.is_file():
            try:
                series = _normalize_history(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, OSError) as exc:
                print(f"  ! Could not read {path.name}: {exc}")
                continue
            if not series:
                continue
            epochs = max(len(v) for v in series.values())
            best = None
            if "val_accuracy" in series:
                best = int(np.argmax(series["val_accuracy"]))
            elif "accuracy" in series:
                best = int(np.argmax(series["accuracy"]))
            return {"epochs": epochs, "series": series, "best_epoch": best, "source": path.name}, path.name
    return None, None


def load_mnist_test(npz: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if npz:
        with np.load(npz) as data:
            return data["x_test"], data["y_test"]
    (_, _), (x_test, y_test) = _keras().datasets.mnist.load_data()
    return x_test, y_test


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def evaluate(model: Any, x_test: np.ndarray, y_test: np.ndarray, seed: int) -> tuple[dict, list]:
    x = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    probs = np.asarray(model.predict(x, batch_size=512, verbose=0), dtype=np.float64)
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    y = y_test.astype(int)

    accuracy = float((pred == y).mean())
    loss = float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (y, pred), 1)

    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    tp = confusion.diagonal()
    per_class = [
        {
            "digit": d,
            "support": int(support[d]),
            "recall": float(tp[d] / support[d]) if support[d] else 0.0,
            "precision": float(tp[d] / predicted[d]) if predicted[d] else 0.0,
        }
        for d in range(10)
    ]

    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(-conf[wrong])][:12]  # the most confidently wrong first
    misclassified = [
        {
            "index": int(i),
            "true": int(y[i]),
            "pred": int(pred[i]),
            "confidence": float(conf[i]),
            "image": _b64(x_test[i]),
        }
        for i in wrong
    ]

    rng = np.random.default_rng(seed)
    gallery = []
    for d in range(10):
        pool = np.flatnonzero((y == d) & (pred == d) & (conf > 0.95))
        if pool.size == 0:
            pool = np.flatnonzero(y == d)
        if pool.size:
            i = int(rng.choice(pool))
            gallery.append({"digit": d, "confidence": float(conf[i]), "image": _b64(x_test[i])})

    evaluation = {
        "samples": int(len(y)),
        "accuracy": accuracy,
        "loss": loss,
        "confusion": confusion.tolist(),
        "per_class": per_class,
        "misclassified": misclassified,
    }
    return evaluation, gallery


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
def print_architecture(model: Any) -> None:
    print("\n  Layer                      Output shape        Params")
    print("  " + "-" * 52)
    for layer in model.layers:
        if layer.__class__.__name__ == "InputLayer":
            continue
        try:
            shape = "x".join(str(d) for d in layer.output.shape[1:])
        except Exception:  # noqa: BLE001
            shape = "?"
        label = f"{layer.name} ({layer.__class__.__name__})"
        print(f"  {label:<26} {shape:<18} {layer.count_params():>8,}")
    print("  " + "-" * 52)
    print(f"  Total parameters: {model.count_params():,}\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Build visualizer data for the Digit Reader.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to the .keras model")
    parser.add_argument("--history", type=Path, help="Training history JSON (Keras history.history)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Where to write viz_data.json")
    parser.add_argument("--mnist-npz", type=Path, help="Local mnist.npz with x_test and y_test (skips the download)")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N test images")
    parser.add_argument("--no-eval", action="store_true", help="Skip the test-set evaluation")
    parser.add_argument("--seed", type=int, default=7, help="Seed for choosing the sample digits")
    args = parser.parse_args()

    started = time.time()
    print(f"\n  Loading model: {args.model}")
    model = load_model(args.model)
    print_architecture(model)

    training, history_name = load_history(args.history)
    if training:
        best = training["best_epoch"]
        extra = f", best epoch {best + 1}" if best is not None else ""
        print(f"  Training history: {history_name} ({training['epochs']} epochs{extra})")
    else:
        print("  Training history: not found. Training curves will be hidden.")
        print('    Save it from your notebook with: json.dump(history.history, open("history.json", "w"))')
        print("    then run:  python visualize.py --history history.json")

    evaluation, gallery = None, None
    if not args.no_eval:
        print("  Evaluating on the MNIST test set...")
        try:
            x_test, y_test = load_mnist_test(args.mnist_npz)
            if args.limit:
                x_test, y_test = x_test[: args.limit], y_test[: args.limit]
            evaluation, gallery = evaluate(model, x_test, y_test, args.seed)
            print(
                f"  Test accuracy: {evaluation['accuracy'] * 100:.2f}% "
                f"({evaluation['samples']:,} images), loss {evaluation['loss']:.4f}"
            )
        except Exception as exc:  # noqa: BLE001 - offline, missing data, etc.
            print(f"  ! Evaluation skipped: {exc}")
            print("    Offline? Download mnist.npz once and pass it with --mnist-npz.")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training": training,
        "evaluation": evaluation,
        "gallery": gallery,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size_kb = args.out.stat().st_size / 1024
    print(f"\n  Wrote {args.out} ({size_kb:.0f} KB) in {time.time() - started:.1f}s")
    print("  Next: uvicorn main:app --reload   then open http://localhost:8000/visualize\n")


if __name__ == "__main__":
    main()
