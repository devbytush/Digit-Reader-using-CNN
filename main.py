"""Handwritten Digit to Text API.

Send a photo of a single handwritten digit and get it back as text
(e.g. "7" / "seven") along with a confidence score and runner-up guesses.

The API is JSON-only. Any frontend (see index.html) can call it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, AsyncIterator

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger("digit2text")

APP_DIR = Path(__file__).resolve().parent
DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


# --------------------------------------------------------------------------- #
# Configuration. Override with env vars prefixed D2T_, e.g.
#   D2T_MODEL_PATH=/models/mnist_cnn.keras
#   D2T_CORS_ORIGINS='["https://my-frontend.example.com"]'   (JSON list)
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="D2T_")

    model_path: Path = APP_DIR / "exported_model" / "mnist_cnn.keras"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    # Minimum share of the image that must be ink, otherwise we treat it as blank.
    min_ink_ratio: float = 0.001
    low_confidence_threshold: float = 0.70


settings = Settings()
Image.MAX_IMAGE_PIXELS = settings.max_image_pixels


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    digit: int = Field(ge=0, le=9)
    probability: float = Field(ge=0, le=1)


class DigitResponse(BaseModel):
    text: str = Field(description='The recognized digit as text, e.g. "7".')
    digit: int = Field(ge=0, le=9)
    word: str = Field(description='The digit spelled out, e.g. "seven".')
    confidence: float = Field(ge=0, le=1)
    top_predictions: list[Candidate] = Field(description="Top 3 candidates, best first.")
    warnings: list[str] = []


class StatusResponse(BaseModel):
    status: str


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
@dataclass
class DigitClassifier:
    model: Any
    lock: Lock = field(default_factory=Lock)

    @classmethod
    def load(cls, path: Path) -> "DigitClassifier":
        if not path.is_file():
            raise RuntimeError(f"Model file not found: {path}")
        from tensorflow import keras  # lazy: heavy import

        return cls(model=keras.models.load_model(path))

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        # Keras inference isn't guaranteed thread-safe; serialize access.
        with self.lock:
            return np.asarray(self.model.predict(tensor, verbose=0)[0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _image_error(status: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status, detail=detail)


def _otsu_threshold(gray: np.ndarray) -> int:
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


def _decode_image(contents: bytes) -> np.ndarray:
    """Bytes -> upright grayscale uint8 array (transparency flattened onto white)."""
    try:
        with Image.open(BytesIO(contents)) as source:
            if source.width * source.height > settings.max_image_pixels:
                raise _image_error(413, "Image is too large. Use an image under 20 megapixels.")
            oriented = ImageOps.exif_transpose(source).convert("RGBA")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise _image_error(415, "The upload is not a supported or readable image.") from exc

    background = Image.new("RGBA", oriented.size, (255, 255, 255, 255))
    return np.asarray(Image.alpha_composite(background, oriented).convert("L"), dtype=np.uint8)


def _to_light_on_dark(gray: np.ndarray) -> np.ndarray:
    """MNIST is bright strokes on a dark background; invert dark-on-light photos."""
    border = np.concatenate((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]))
    return 255 - gray if float(np.median(border)) >= 127 else gray


def _column_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Horizontal runs of columns containing foreground, split by empty columns."""
    occupied = np.concatenate(([0], mask.any(axis=0).astype(np.int8), [0]))
    edges = np.diff(occupied)
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _center_on_canvas(glyph: np.ndarray) -> np.ndarray:
    """Fit into 20x20, place on 28x28, then center by center of mass (MNIST style)."""
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


def _prepare_tensor(contents: bytes) -> np.ndarray:
    gray = _to_light_on_dark(_decode_image(contents))
    foreground = gray > _otsu_threshold(gray)

    runs = _column_runs(foreground)
    if not runs:
        raise _image_error(422, "No handwritten digit was found. Try a clearer, higher-contrast photo.")

    # Camera shots often contain stray specks/shadows: keep only the run with the most ink.
    left, right = max(runs, key=lambda r: int(foreground[:, r[0] : r[1]].sum()))
    rows = np.flatnonzero(foreground[:, left:right].any(axis=1))
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    mask = foreground[top:bottom, left:right]

    if mask.sum() < foreground.size * settings.min_ink_ratio:
        raise _image_error(422, "The digit is too small or faint. Move closer and try again.")

    # Keep original stroke intensity for smoother, MNIST-like edges.
    glyph = np.where(mask, gray[top:bottom, left:right], 0).astype(np.uint8)
    return _center_on_canvas(glyph).reshape(1, 28, 28, 1)


# --------------------------------------------------------------------------- #
# Recognition
# --------------------------------------------------------------------------- #
def _recognize(contents: bytes, classifier: DigitClassifier) -> DigitResponse:
    probabilities = classifier.predict(_prepare_tensor(contents))
    ranked = np.argsort(probabilities)[::-1][:3]
    digit = int(ranked[0])
    confidence = float(probabilities[digit])

    warnings = []
    if confidence < settings.low_confidence_threshold:
        warnings.append("Low confidence. Try a clearer photo with one large, dark digit.")

    return DigitResponse(
        text=str(digit),
        digit=digit,
        word=DIGIT_WORDS[digit],
        confidence=confidence,
        top_predictions=[
            Candidate(digit=int(i), probability=float(probabilities[i])) for i in ranked
        ],
        warnings=warnings,
    )


async def _read_limited(file: UploadFile) -> bytes:
    try:
        contents = await file.read(settings.max_upload_bytes + 1)
    finally:
        await file.close()
    if not contents:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"Image is too large. Maximum upload size is {settings.max_upload_bytes // (1024 * 1024)} MB."
        )
    return contents


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load once at startup: no slow first request, and a missing model fails fast.
    logger.info("Loading model from %s", settings.model_path)
    app.state.classifier = await run_in_threadpool(DigitClassifier.load, settings.model_path)
    yield
    app.state.classifier = None


app = FastAPI(
    title="Handwritten Digit to Text API",
    description="Convert a photo of a single handwritten digit (0-9) into text.",
    version="2.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.rstrip("/") for origin in settings.cors_origins],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_classifier(request: Request) -> DigitClassifier:
    classifier: DigitClassifier | None = getattr(request.app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(503, "Model is not loaded.")
    return classifier


@app.get("/", include_in_schema=False)
async def root() -> Response:
    """Serve the frontend (index.html next to this file) if present, else the API docs."""
    page = APP_DIR / "index.html"
    return FileResponse(page) if page.is_file() else RedirectResponse("/docs")


@app.get("/health", response_model=StatusResponse, tags=["system"])
async def health() -> StatusResponse:
    """Liveness probe."""
    return StatusResponse(status="ok")


@app.get("/ready", response_model=StatusResponse, tags=["system"])
async def ready(request: Request) -> StatusResponse:
    """Readiness probe: succeeds once the model is loaded."""
    _get_classifier(request)
    return StatusResponse(status="ready")


@app.post(
    "/v1/convert",
    response_model=DigitResponse,
    tags=["conversion"],
    summary="Convert a handwritten digit image to text",
    responses={
        400: {"description": "Empty upload"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported or unreadable image"},
        422: {"description": "No digit found"},
        503: {"description": "Model not loaded"},
    },
)
async def convert(
    request: Request,
    file: Annotated[UploadFile, File(description="Image containing one handwritten digit")],
) -> DigitResponse:
    classifier = _get_classifier(request)
    contents = await _read_limited(file)
    return await run_in_threadpool(_recognize, contents, classifier)
