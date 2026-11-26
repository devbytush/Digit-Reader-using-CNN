"""Photo -> MNIST-style 28x28 tensor.

Pure functions with no web-framework dependency, so they can be unit-tested and
evaluated offline (see ``eval/pipeline_eval.py``). The contract with the model is
documented in ``exported_model/preprocessing_config.json``: float32, shape
(1, 28, 28, 1), values in [0, 1], bright digit on a dark background, digit fitted
to a 20x20 box and centred by centre of mass (the way MNIST was prepared).
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

DEFAULT_MAX_IMAGE_PIXELS = 20_000_000
DEFAULT_MIN_INK_RATIO = 0.001


class PreprocessError(Exception):
    """A user-facing image problem. ``status_code`` mirrors the HTTP status the API returns."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / gray.size
    weight_bg = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * np.arange(256))
    denominator = weight_bg * (1.0 - weight_bg)
    variance = np.zeros(256)
    valid = denominator > 0
    variance[valid] = (
        (cumulative_mean[-1] * weight_bg[valid] - cumulative_mean[valid]) ** 2
        / denominator[valid]
    )
    return int(np.argmax(variance))


def decode_image(contents: bytes, max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS) -> np.ndarray:
    """Bytes -> upright grayscale uint8 array (transparency flattened onto white)."""
    try:
        with Image.open(BytesIO(contents)) as source:
            if source.width * source.height > max_pixels:
                raise PreprocessError(413, "Image is too large. Use an image under 20 megapixels.")
            oriented = ImageOps.exif_transpose(source).convert("RGBA")
    except PreprocessError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise PreprocessError(415, "The upload is not a supported or readable image.") from exc

    background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(background, oriented).convert("L"), dtype=np.uint8)


def to_light_on_dark(gray: np.ndarray) -> np.ndarray:
    """MNIST is bright strokes on a dark background; invert dark-on-light photos."""
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    return 255 - gray if float(np.median(border)) >= 127 else gray


def column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Horizontal runs of columns containing foreground, split by empty columns."""
    occupied = np.concatenate(([0], mask.any(axis=0).astype(np.int8), [0]))
    edges = np.diff(occupied)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def center_on_canvas(glyph: np.ndarray) -> np.ndarray:
    """Fit into 20x20, place on 28x28, then centre by centre of mass (MNIST style)."""
    height, width = glyph.shape
    scale = 20.0 / max(height, width)
    resized = Image.fromarray(glyph).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - resized.height) // 2, (28 - resized.width) // 2
    canvas[top : top + resized.height, left : left + resized.width] = np.asarray(
        resized, dtype=np.float32
    )

    total = canvas.sum()
    if total > 0:
        ys, xs = np.indices(canvas.shape)
        shift_y = int(round(13.5 - (ys * canvas).sum() / total))
        shift_x = int(round(13.5 - (xs * canvas).sum() / total))
        canvas = np.roll(canvas, (shift_y, shift_x), axis=(0, 1))
    return canvas / 255.0


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Remove slowly varying illumination from a light-on-dark image (stroke bright, page dark).

    The page level is estimated on a small copy with a minimum filter wider than any stroke,
    then smoothed and subtracted, so a shadow gradient no longer masquerades as ink.
    """
    height, width = gray.shape
    scale = min(1.0, 128.0 / max(height, width))
    small = Image.fromarray(gray).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR
    )
    window = max(3, round(0.18 * max(small.size)) | 1)  # odd, comfortably wider than a stroke
    page = small.filter(ImageFilter.MinFilter(window)).filter(ImageFilter.GaussianBlur(window / 3))
    page = np.asarray(page.resize((width, height), Image.Resampling.BILINEAR), dtype=np.int16)
    return np.clip(gray.astype(np.int16) - page, 0, 255).astype(np.uint8)


def smooth_for_segmentation(gray: np.ndarray) -> np.ndarray:
    """Light blur (scaled to image size) so isolated noise specks do not become 'ink'."""
    sigma = max(1.0, 0.004 * max(gray.shape))
    return np.asarray(Image.fromarray(gray).filter(ImageFilter.GaussianBlur(sigma)), dtype=np.uint8)


def prepare_from_gray(
    gray: np.ndarray,
    *,
    min_ink_ratio: float = DEFAULT_MIN_INK_RATIO,
    stretch_contrast: bool = False,
    robust: bool = False,
) -> np.ndarray:
    """Upright grayscale uint8 image -> float32 tensor of shape (1, 28, 28, 1).

    ``stretch_contrast`` rescales the stroke so its brightest pixel is 255, as in MNIST.
    It is off by default (the production behaviour); the evaluation harness measures
    whether turning it on helps. ``robust`` flattens uneven lighting and smooths noise before
    locating the digit; the pixels that reach the model still come from the original image.
    """
    gray = to_light_on_dark(gray)
    if robust:
        gray = flatten_background(gray)
        segmentation = smooth_for_segmentation(gray)
    else:
        segmentation = gray
    foreground = segmentation > otsu_threshold(segmentation)

    runs = column_runs(foreground)
    if not runs:
        raise PreprocessError(422, "No handwritten digit was found. Try a clearer, higher-contrast photo.")

    # Camera shots often contain stray specks/shadows: keep only the run with the most ink.
    left, right = max(runs, key=lambda r: int(foreground[:, r[0] : r[1]].sum()))
    rows = np.flatnonzero(foreground[:, left:right].any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    mask = foreground[top:bottom, left:right]

    if mask.sum() < foreground.size * min_ink_ratio:
        raise PreprocessError(422, "The digit is too small or faint. Move closer and try again.")

    # Keep original stroke intensity for smoother, MNIST-like edges.
    glyph = np.where(mask, gray[top:bottom, left:right], 0).astype(np.uint8)
    if stretch_contrast and glyph.max() > 0:
        glyph = np.rint(glyph.astype(np.float32) * (255.0 / glyph.max())).astype(np.uint8)
    return center_on_canvas(glyph).reshape(1, 28, 28, 1)


def prepare_tensor(
    contents: bytes,
    *,
    max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    min_ink_ratio: float = DEFAULT_MIN_INK_RATIO,
    stretch_contrast: bool = False,
    robust: bool = False,
) -> np.ndarray:
    """Encoded image bytes -> model-ready tensor. Raises :class:`PreprocessError`."""
    return prepare_from_gray(
        decode_image(contents, max_pixels),
        min_ink_ratio=min_ink_ratio,
        stretch_contrast=stretch_contrast,
        robust=robust,
    )
