"""Handwritten Digit to Text API (with a live network-visualizer backend).

Send a photo of a single handwritten digit and get it back as text
(e.g. "7" / "seven") along with a confidence score and runner-up guesses.

The API is JSON-only. Any frontend (see index.html) can call it.
"""

from __future__ import annotations

import base64
import json
import logging
import time as _time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, AsyncIterator
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from digitreader.preprocessing import PreprocessError, prepare_tensor

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
    # Background flattening + noise smoothing + contrast stretch before locating the digit.
    # Measured in eval/results/pipeline_eval_variants.json. Set D2T_ROBUST_PREPROCESSING=false
    # to restore the original pipeline without redeploying code.
    robust_preprocessing: bool = True
    low_confidence_threshold: float = 0.70
    # Written by visualize.py: training curves, evaluation, sample digits.
    viz_data_path: Path = APP_DIR / "exported_model" / "viz_data.json"


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


class Base64Request(BaseModel):
    image: str = Field(description="Base64-encoded image data (data URI or raw base64 string)")


class MetricsResponse(BaseModel):
    model_name: str
    total_params: int
    uptime_seconds: float = Field(description="Seconds since the server started")
    total_requests: int
    model_loaded: bool
    input_shape: list[int]


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
def _keras() -> Any:
    try:
        import keras  # standalone Keras 3 (installed with TensorFlow >= 2.16)

        return keras
    except ImportError:
        from tensorflow import keras  # TensorFlow 2.15 and older

        return keras


@dataclass
class DigitClassifier:
    model: Any
    lock: Lock = field(default_factory=Lock)
    _extractor: Any = field(default=None, init=False, repr=False)

    @classmethod
    def load(cls, path: Path) -> "DigitClassifier":
        if not path.is_file():
            raise RuntimeError(f"Model file not found: {path}")
        return cls(model=_keras().models.load_model(path))

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        # Keras inference isn't guaranteed thread-safe; serialize access.
        with self.lock:
            return np.asarray(self.model.predict(tensor, verbose=0)[0], dtype=np.float64)

    def layers(self) -> list[Any]:
        """Real layers in order, without the Input layer."""
        return [layer for layer in self.model.layers if layer.__class__.__name__ != "InputLayer"]

    def trace(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Run one image and return the output of every layer (batch dimension included)."""
        with self.lock:
            layers = self.layers()
            if self._extractor is None:
                try:
                    self._extractor = _keras().Model(self.model.input, [layer.output for layer in layers])
                except Exception:  # noqa: BLE001 - fall back to a plain layer-by-layer pass
                    self._extractor = False
            if self._extractor:
                outputs = self._extractor(tensor, training=False)
                outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
                return [np.asarray(o) for o in outputs]
            x, outputs = tensor, []
            for layer in layers:
                x = layer(x, training=False)
                outputs.append(np.asarray(x))
            return outputs


# --------------------------------------------------------------------------- #
# Preprocessing lives in digitreader/preprocessing.py (pure, unit-tested).
# --------------------------------------------------------------------------- #
def _prepare_tensor(contents: bytes) -> np.ndarray:
    try:
        return prepare_tensor(
            contents,
            max_pixels=settings.max_image_pixels,
            min_ink_ratio=settings.min_ink_ratio,
            robust=settings.robust_preprocessing,
            stretch_contrast=settings.robust_preprocessing,
        )
    except PreprocessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


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


def _max_b64_chars() -> int:
    """Longest base64 string that can decode to an allowed upload (4 chars per 3 bytes, plus a data-URI prefix)."""
    return (settings.max_upload_bytes * 4) // 3 + 1024


def _websocket_origin_allowed(ws: WebSocket) -> bool:
    """CORS middleware does not cover WebSockets, so check the Origin header ourselves.

    Non-browser clients send no Origin and are allowed. Browsers must come from the same
    host that serves this app or from a configured CORS origin.
    """
    origin = ws.headers.get("origin")
    if not origin:
        return True
    allowed = {o.rstrip("/") for o in settings.cors_origins}
    return origin.rstrip("/") in allowed or urlparse(origin).netloc == ws.headers.get("host", "")


# --------------------------------------------------------------------------- #
# Network visualizer: architecture + live layer-by-layer trace
# --------------------------------------------------------------------------- #
def _b64(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _quantize(array: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Scale to 0..255 across the whole layer so brightness stays comparable between channels."""
    lo, hi = float(array.min()), float(array.max())
    span = hi - lo
    if span < 1e-9:
        return lo, hi, np.zeros(array.shape, dtype=np.uint8)
    return lo, hi, np.rint((array - lo) / span * 255).astype(np.uint8)


def _plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _layer_config(layer: Any) -> dict[str, Any]:
    config = layer.get_config()
    keep = ("filters", "kernel_size", "strides", "padding", "pool_size", "units", "rate", "axis", "momentum")
    out = {key: _plain(config[key]) for key in keep if key in config}
    activation = config.get("activation")
    if isinstance(activation, dict):
        activation = activation.get("config", {}).get("name") or activation.get("class_name")
    if activation is not None:
        out["activation"] = str(activation)
    return out


def _first_conv_kernels(layers: list[Any]) -> list[list[list[float]]] | None:
    """Kernels of the first Conv2D layer, averaged over input channels: [filters][rows][cols]."""
    for layer in layers:
        if layer.__class__.__name__ == "Conv2D":
            weights = layer.get_weights()
            if weights and weights[0].ndim == 4:
                kernels = np.moveaxis(weights[0].mean(axis=2), -1, 0)  # (filters, kh, kw)
                return np.round(kernels, 4).tolist()
    return None


def describe_model(classifier: DigitClassifier) -> dict[str, Any]:
    """Static architecture description, built once at startup from a zero image."""
    model = classifier.model
    try:
        input_shape = [int(d) for d in model.input_shape[1:]]
    except Exception:  # noqa: BLE001
        input_shape = [28, 28, 1]
    layers = classifier.layers()
    outputs = classifier.trace(np.zeros((1, *input_shape), dtype=np.float32))
    return {
        "name": str(model.name),
        "input_shape": input_shape,
        "total_params": int(model.count_params()),
        "layers": [
            {
                "index": i,
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": [int(d) for d in out.shape[1:]],
                "params": int(layer.count_params()),
                "config": _layer_config(layer),
            }
            for i, (layer, out) in enumerate(zip(layers, outputs))
        ],
        "kernels": _first_conv_kernels(layers),
    }


def _load_viz_data() -> dict[str, Any]:
    """Training history, evaluation and sample digits written by visualize.py (optional)."""
    empty: dict[str, Any] = {"training": None, "evaluation": None, "gallery": None}
    path = settings.viz_data_path
    if not path.is_file():
        return empty
    try:
        return {**empty, **json.loads(path.read_text(encoding="utf-8"))}
    except (OSError, ValueError):
        logger.warning("Could not read %s", path)
        return empty


def _trace(contents: bytes, classifier: DigitClassifier) -> dict[str, Any]:
    tensor = _prepare_tensor(contents)
    outputs = classifier.trace(tensor)

    layers = []
    for layer, out in zip(classifier.layers(), outputs):
        array = np.asarray(out[0], dtype=np.float32)
        lo, hi, quantized = _quantize(array)
        if array.ndim == 3:  # (H, W, C) -> channel-first so each channel is one contiguous plane
            quantized = np.moveaxis(quantized, -1, 0)
        entry: dict[str, Any] = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "shape": list(array.shape),
            "layout": "chw" if array.ndim == 3 else "flat",
            "min": lo,
            "max": hi,
            "mean": float(array.mean()),
            "data": _b64(quantized),
        }
        if array.size <= 64:
            entry["values"] = [round(float(v), 5) for v in array.ravel()]
        layers.append(entry)

    probabilities = np.asarray(outputs[-1][0], dtype=np.float64)
    ranked = np.argsort(probabilities)[::-1][:3]
    digit = int(ranked[0])
    return {
        "prediction": digit,
        "word": DIGIT_WORDS[digit] if digit < 10 else str(digit),
        "confidence": float(probabilities[digit]),
        "probabilities": [round(float(p), 6) for p in probabilities],
        "top_predictions": [
            {"digit": int(i), "probability": float(probabilities[i])} for i in ranked
        ],
        "input": {"shape": [28, 28], "data": _b64(np.rint(tensor[0, :, :, 0] * 255).astype(np.uint8))},
        "layers": layers,
    }


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Load once at startup: no slow first request, and a missing model fails fast.
    logger.info("Loading model from %s", settings.model_path)
    app.state.classifier = await run_in_threadpool(DigitClassifier.load, settings.model_path)
    app.state.architecture = await run_in_threadpool(describe_model, app.state.classifier)
    yield
    app.state.classifier = None
    app.state.architecture = None


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


# --------------------------------------------------------------------------- #
# Request lifecycle middleware
# --------------------------------------------------------------------------- #
_boot_time = _time.monotonic()
_request_count = 0


@app.middleware("http")
async def request_lifecycle(request: Request, call_next):
    global _request_count
    _request_count += 1
    request_id = uuid.uuid4().hex[:8]
    start = _time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((_time.monotonic() - start) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    if request.url.path == "/v1/model":
        response.headers["Cache-Control"] = "public, max-age=300"
    return response


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


@app.get("/visualize", include_in_schema=False)
@app.get("/visualize.html", include_in_schema=False)
async def visualize_page() -> Response:
    """The 3D network visualizer (visualize.html next to this file)."""
    page = APP_DIR / "visualize.html"
    if not page.is_file():
        raise HTTPException(404, "visualize.html was not found next to main.py.")
    return FileResponse(page)


FRONTEND_DIST = APP_DIR / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/react/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="react_assets")


@app.get("/react", include_in_schema=False)
@app.get("/react/", include_in_schema=False)
@app.get("/app", include_in_schema=False)
async def react_app() -> Response:
    """Serve the modern React + GSAP + Lenis frontend."""
    dist_page = FRONTEND_DIST / "index.html"
    if dist_page.is_file():
        return FileResponse(dist_page)
    raise HTTPException(404, "React frontend build not found. Run 'npm run build' inside frontend/")


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


@app.get("/v1/model", tags=["visualizer"], summary="Architecture plus training data for the visualizer")
async def model_info(request: Request) -> dict[str, Any]:
    _get_classifier(request)
    architecture = getattr(request.app.state, "architecture", None)
    if architecture is None:
        raise HTTPException(503, "Model is not loaded.")
    return {**architecture, **_load_viz_data()}


@app.post(
    "/v1/trace",
    tags=["visualizer"],
    summary="Run an image and return every layer's activations",
    responses={
        413: {"description": "Image too large"},
        415: {"description": "Unsupported or unreadable image"},
        422: {"description": "No digit found"},
    },
)
async def trace(
    request: Request,
    file: Annotated[UploadFile, File(description="Image containing one handwritten digit")],
) -> dict[str, Any]:
    classifier = _get_classifier(request)
    contents = await _read_limited(file)
    return await run_in_threadpool(_trace, contents, classifier)


# --------------------------------------------------------------------------- #
# Additional endpoints
# --------------------------------------------------------------------------- #
@app.post(
    "/v1/convert-base64",
    response_model=DigitResponse,
    tags=["conversion"],
    summary="Convert a base64-encoded handwritten digit image to text",
    responses={
        400: {"description": "Invalid or empty base64"},
        413: {"description": "Image too large"},
        415: {"description": "Unsupported or unreadable image"},
        422: {"description": "No digit found"},
        503: {"description": "Model not loaded"},
    },
)
async def convert_base64(request: Request, body: Base64Request) -> DigitResponse:
    classifier = _get_classifier(request)
    raw = body.image.strip()
    if len(raw) > _max_b64_chars():
        raise HTTPException(413, f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.")
    # Strip the data-URI prefix if the client sent one (e.g. "data:image/png;base64,...")
    if "," in raw and raw.index(",") < 120:
        raw = raw.split(",", 1)[1]
    try:
        contents = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise HTTPException(400, "Invalid base64 image data.") from exc
    if not contents:
        raise HTTPException(400, "The decoded image data is empty.")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            413,
            f"Image exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
        )
    return await run_in_threadpool(_recognize, contents, classifier)


@app.get(
    "/v1/metrics",
    response_model=MetricsResponse,
    tags=["system"],
    summary="Model and server metrics",
)
async def metrics(request: Request) -> MetricsResponse:
    classifier = getattr(request.app.state, "classifier", None)
    architecture = getattr(request.app.state, "architecture", None)
    return MetricsResponse(
        model_name=architecture["name"] if architecture else "unknown",
        total_params=architecture["total_params"] if architecture else 0,
        uptime_seconds=round(_time.monotonic() - _boot_time, 1),
        total_requests=_request_count,
        model_loaded=classifier is not None,
        input_shape=architecture["input_shape"] if architecture else [28, 28, 1],
    )


@app.websocket("/ws/draw")
async def ws_draw(ws: WebSocket) -> None:
    """Real-time digit recognition over WebSocket.

    Send ``{"image": "<base64>"}`` and receive the full ``DigitResponse`` as
    JSON.  The *image* value may be a raw base64 string or a data-URI.
    """
    if not _websocket_origin_allowed(ws):
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()
    classifier = getattr(ws.app.state, "classifier", None)
    if classifier is None:
        await ws.close(1011, "Model is not loaded.")
        return
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > _max_b64_chars() + 64:  # +64 for the JSON envelope
                await ws.send_json({"error": "Image is too large."})
                continue
            try:
                payload = json.loads(raw)
                image_data = payload.get("image", "")
                if "," in image_data and image_data.index(",") < 120:
                    image_data = image_data.split(",", 1)[1]
                contents = base64.b64decode(image_data, validate=True)
                result = await run_in_threadpool(_recognize, contents, classifier)
                await ws.send_json(result.model_dump())
            except HTTPException as exc:
                await ws.send_json({"error": exc.detail})
            except Exception as exc:  # noqa: BLE001
                await ws.send_json({"error": f"Processing error: {exc}"})
    except WebSocketDisconnect:
        pass
