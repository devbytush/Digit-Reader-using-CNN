import base64

import pytest
from starlette.websockets import WebSocketDisconnect


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_every_real_mnist_digit_is_read_correctly(client, mnist_samples, photo):
    """End to end: HTTP upload -> preprocessing -> real CNN, on one real digit per class."""
    for digit, glyph in mnist_samples:
        response = client.post("/v1/convert", files={"file": ("d.png", photo(glyph), "image/png")})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["digit"] == digit
        assert body["text"] == str(digit)
        assert body["confidence"] > 0.9
        assert len(body["top_predictions"]) == 3


@pytest.mark.parametrize(
    ("files", "status"),
    [
        ({"file": ("e.png", b"", "image/png")}, 400),
        ({"file": ("t.txt", b"hello", "text/plain")}, 415),
    ],
)
def test_bad_uploads(client, files, status):
    assert client.post("/v1/convert", files=files).status_code == status


def test_blank_image_returns_422(client):
    import io

    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64), 255, np.uint8)).save(buffer, format="PNG")
    response = client.post("/v1/convert", files={"file": ("b.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 422


def test_base64_endpoint_matches_multipart(client, mnist_samples, photo):
    digit, glyph = mnist_samples[7]
    payload = base64.b64encode(photo(glyph)).decode()
    response = client.post("/v1/convert-base64", json={"image": f"data:image/png;base64,{payload}"})
    assert response.status_code == 200 and response.json()["digit"] == digit


def test_base64_oversize_is_rejected_before_decoding(client, monkeypatch):
    import main

    monkeypatch.setattr(main.settings, "max_upload_bytes", 1000)
    assert client.post("/v1/convert-base64", json={"image": "A" * 5000}).status_code == 413


def test_model_description_matches_training_summary(client):
    import json

    from conftest import ROOT

    summary = json.loads((ROOT / "exported_model" / "training_summary.json").read_text(encoding="utf-8"))
    info = client.get("/v1/model").json()
    assert info["total_params"] == summary["total_parameters"]
    assert info["input_shape"] == [28, 28, 1]
    assert [layer["type"] for layer in info["layers"]][-1] == "Dense"


def test_trace_agrees_with_convert(client, mnist_samples, photo):
    digit, glyph = mnist_samples[2]
    files = {"file": ("d.png", photo(glyph), "image/png")}
    trace = client.post("/v1/trace", files=files).json()
    assert trace["prediction"] == digit
    assert abs(sum(trace["probabilities"]) - 1.0) < 1e-4
    assert len(trace["layers"]) == len(client.get("/v1/model").json()["layers"])


def test_websocket_same_origin_and_no_origin_allowed(client, mnist_samples, photo):
    digit, glyph = mnist_samples[5]
    message = {"image": base64.b64encode(photo(glyph)).decode()}
    with client.websocket_connect("/ws/draw") as ws:
        ws.send_json(message)
        assert ws.receive_json()["digit"] == digit


def test_websocket_foreign_origin_is_rejected(client):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/draw", headers={"origin": "https://evil.example"}):
        pass


def test_websocket_oversize_message_is_refused(client, monkeypatch):
    import main

    monkeypatch.setattr(main.settings, "max_upload_bytes", 1000)
    with client.websocket_connect("/ws/draw") as ws:
        ws.send_json({"image": "A" * 6000})
        assert "too large" in ws.receive_json()["error"].lower()
