import base64
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def mnist_samples() -> list[tuple[int, np.ndarray]]:
    """Ten real MNIST test digits (28x28 uint8) shipped in exported_model/viz_data.json."""
    data = json.loads((ROOT / "exported_model" / "viz_data.json").read_text(encoding="utf-8"))
    return [
        (item["digit"], np.frombuffer(base64.b64decode(item["image"]), dtype=np.uint8).reshape(28, 28))
        for item in data["gallery"]
    ]


def render_photo(glyph28: np.ndarray, *, size: int = 280, dark_on_light: bool = True, pad: int = 60) -> bytes:
    """Upscale an MNIST digit and render it like a clean phone photo of ink on paper (PNG bytes)."""
    big = np.asarray(Image.fromarray(glyph28).resize((size, size), Image.Resampling.BICUBIC))
    canvas = np.zeros((size + 2 * pad, size + 2 * pad), dtype=np.uint8)
    canvas[pad : pad + size, pad : pad + size] = big
    if dark_on_light:
        canvas = 255 - canvas
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def photo():
    return render_photo


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as test_client:  # runs lifespan: loads the real model once
        yield test_client
