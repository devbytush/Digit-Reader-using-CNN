#!/usr/bin/env python3
"""Measure end-to-end accuracy of the *camera pipeline*, not just the bare CNN.

The 98.96% in exported_model/training_summary.json is accuracy on clean 28x28 MNIST. Real
users upload photos, so this harness renders MNIST *test* digits as simulated photos
(ink on paper, blur, noise, shadows, rotation, faint or thick strokes, JPEG, ...), pushes
them through ``digitreader.preprocessing`` and the real model, and reports accuracy per
condition with Wilson 95% confidence intervals.

It also compares preprocessing variants on the *same* rendered images (paired design) and
reports an exact McNemar p-value, so a change is only adopted when the data supports it.

    python eval/pipeline_eval.py --mnist-npz data/mnist_test.npz --n 2000

Limits, stated plainly: the photos are simulated, not real phone captures, and the digits
are MNIST test digits (the model never trained on them). Treat results as a relative
comparison between pipeline variants and a stress test, not as a field accuracy guarantee.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from digitreader.preprocessing import PreprocessError, prepare_from_gray, prepare_tensor  # noqa: E402


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    size: tuple[int, int] = (120, 260)  # digit side length in pixels, on a 400 px canvas
    rotate: float = 0.0  # max abs rotation in degrees
    ink: float = 1.0  # stroke contrast: 1 = near-black ink, 0.3 = faint pencil
    blur: float = 0.0  # gaussian radius
    noise: float = 0.0  # gaussian sigma (grey levels)
    shadow: float = 0.0  # illumination gradient strength across the page
    thicken: int = 0  # MaxFilter size (thick marker); 0 = off
    thin: int = 0  # MinFilter size (thin pen); 0 = off
    jpeg: int = 0  # JPEG quality; 0 = off (exercises the decode path too)
    light_on_dark: bool = False  # canvas-style drawing (white stroke on black)


CONDITIONS = [
    Condition("clean_photo", "dark ink on white paper, no degradation"),
    Condition("light_on_dark", "white stroke on black (drawing canvas)", light_on_dark=True),
    Condition("faint_ink", "pencil-like, 35% contrast", ink=0.35),
    Condition("very_faint_ink", "20% contrast", ink=0.20),
    Condition("blur", "defocus blur, radius 3", blur=3.0),
    Condition("noise", "sensor noise, sigma 20", noise=20.0),
    Condition("rotated", "rotated up to +/-15 degrees", rotate=15.0),
    Condition("shadow", "uneven lighting, 50% gradient", shadow=0.5),
    Condition("small_digit", "digit only 40-70 px on a 400 px frame", size=(40, 70)),
    Condition("thick_marker", "thick strokes", thicken=11),
    Condition("thin_pen", "thin strokes", thin=5),
    Condition("jpeg_q30", "heavy JPEG compression, quality 30", jpeg=30),
    Condition(
        "combined_hard",
        "faint+blur+noise+shadow+rotation",
        ink=0.5,
        blur=2.0,
        noise=12.0,
        shadow=0.35,
        rotate=10.0,
    ),
]


def render(glyph: np.ndarray, cond: Condition, rng: np.random.Generator, canvas: int = 400) -> np.ndarray:
    """MNIST 28x28 uint8 -> simulated photo, upright grayscale uint8."""
    side = int(rng.integers(cond.size[0], cond.size[1] + 1))
    big = Image.fromarray(glyph).resize((side, side), Image.Resampling.BICUBIC)
    if cond.thicken:
        big = big.filter(ImageFilter.MaxFilter(cond.thicken))
    if cond.thin:
        big = big.filter(ImageFilter.MinFilter(cond.thin))
    if cond.rotate:
        big = big.rotate(float(rng.uniform(-cond.rotate, cond.rotate)), resample=Image.Resampling.BICUBIC)
    frame = np.zeros((canvas, canvas), dtype=np.float32)
    top = int(rng.integers(0, canvas - side + 1))
    left = int(rng.integers(0, canvas - side + 1))
    frame[top : top + side, left : left + side] = np.asarray(big, dtype=np.float32) / 255.0

    if cond.light_on_dark:
        photo = frame * 255.0 * cond.ink
    else:
        paper = float(rng.uniform(205, 245))
        photo = paper - frame * cond.ink * (paper - 25.0)
    if cond.shadow:
        angle = float(rng.uniform(0, 2 * math.pi))
        ys, xs = np.mgrid[0:canvas, 0:canvas] / canvas
        ramp = (xs * math.cos(angle) + ys * math.sin(angle) + 1) / 2
        photo = photo * (1.0 - cond.shadow * ramp)
    image = Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8))
    if cond.blur:
        image = image.filter(ImageFilter.GaussianBlur(cond.blur))
    out = np.asarray(image, dtype=np.float32)
    if cond.noise:
        out = out + rng.normal(0.0, cond.noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


VARIANTS: dict[str, dict] = {
    "current": {},
    "stretch": {"stretch_contrast": True},
    "robust": {"robust": True},
    "robust+stretch": {"robust": True, "stretch_contrast": True},
}


def to_tensor(photo: np.ndarray, cond: Condition, options: dict) -> np.ndarray | None:
    """Run the real pipeline; None means the API would have rejected the image (HTTP 422/415)."""
    try:
        if cond.jpeg:
            buffer = io.BytesIO()
            Image.fromarray(photo).save(buffer, format="JPEG", quality=cond.jpeg)
            return prepare_tensor(buffer.getvalue(), **options)
        return prepare_from_gray(photo, **options)
    except PreprocessError:
        return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p-value for discordant pairs b (A right, B wrong) and c (A wrong, B right)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def predict(model, tensors: list[np.ndarray | None]) -> np.ndarray:
    """Argmax per image; -1 where the pipeline rejected the photo."""
    preds = np.full(len(tensors), -1, dtype=int)
    ok = [i for i, t in enumerate(tensors) if t is not None]
    if ok:
        batch = np.concatenate([tensors[i] for i in ok], axis=0)
        preds[ok] = model.predict(batch, batch_size=512, verbose=0).argmax(axis=1)
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mnist-npz", type=Path, required=True, help="npz with x_test (N,28,28) uint8 and y_test")
    parser.add_argument("--model", type=Path, default=ROOT / "exported_model" / "mnist_cnn.keras")
    parser.add_argument("--n", type=int, default=2000, help="test images per condition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    import keras

    model = keras.models.load_model(args.model)
    with np.load(args.mnist_npz) as data:
        x_test, y_test = data["x_test"], data["y_test"].astype(int)
    pick = np.random.default_rng(args.seed).permutation(len(y_test))[: args.n]
    x, y = x_test[pick], y_test[pick]

    direct = model.predict((x.astype("float32") / 255.0)[..., None], batch_size=512, verbose=0).argmax(axis=1)
    direct_acc = float((direct == y).mean())
    print(f"\nBare CNN on clean 28x28 ({args.n} images): {direct_acc:.4f}\n")

    results = []
    started = time.time()
    for cond in CONDITIONS:
        rng = np.random.default_rng([args.seed, zlib.crc32(cond.name.encode())])
        photos = [render(g, cond, rng) for g in x]
        row: dict = {"condition": cond.name, "description": cond.description}
        correct: dict[str, np.ndarray] = {}
        for variant, options in VARIANTS.items():
            preds = predict(model, [to_tensor(p, cond, options) for p in photos])
            ok = preds == y
            accepted = int((preds != -1).sum())
            lo, hi = wilson(int(ok.sum()), len(y))
            row[variant] = {
                "accuracy": float(ok.mean()),
                "ci95": [lo, hi],
                "rejected": int((preds == -1).sum()),
                "accuracy_on_accepted": float(ok.sum() / accepted) if accepted else None,
            }
            correct[variant] = ok
        row["paired_vs_current"] = {}
        for variant in list(VARIANTS)[1:]:
            b = int((correct["current"] & ~correct[variant]).sum())
            c = int((~correct["current"] & correct[variant]).sum())
            row["paired_vs_current"][variant] = {
                "only_current_right": b,
                "only_variant_right": c,
                "mcnemar_p": mcnemar_exact(b, c),
            }
        results.append(row)
        cells = []
        for variant in VARIANTS:
            cell = f"{row[variant]['accuracy']:.3f}"
            if variant != "current":
                pair = row["paired_vs_current"][variant]
                mark = ""
                if pair["mcnemar_p"] < 0.01:
                    mark = "+" if pair["only_variant_right"] > pair["only_current_right"] else "-"
                cell += mark
            cells.append(cell)
        print(f"{cond.name:<16} " + "  ".join(f"{v}={c}" for v, c in zip(VARIANTS, cells)), flush=True)

    print(f"\nDone in {time.time() - started:.0f}s")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n": args.n, "seed": args.seed, "direct_28x28_accuracy": direct_acc, "results": results}
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
