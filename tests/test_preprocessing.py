import io

import numpy as np
import pytest
from PIL import Image

from digitreader.preprocessing import (
    PreprocessError,
    otsu_threshold,
    prepare_from_gray,
    prepare_tensor,
)


@pytest.mark.parametrize("robust", [False, True])
def test_tensor_contract(mnist_samples, photo, robust):
    """The model contract in exported_model/preprocessing_config.json."""
    for _, glyph in mnist_samples:
        tensor = prepare_tensor(photo(glyph), robust=robust)
        assert tensor.shape == (1, 28, 28, 1)
        assert tensor.dtype == np.float32
        assert 0.0 <= tensor.min() and tensor.max() <= 1.0


@pytest.mark.parametrize("robust", [False, True])
def test_polarity_is_normalised(mnist_samples, photo, robust):
    """Dark-on-light and light-on-dark renderings of one digit give the identical tensor."""
    _, glyph = mnist_samples[3]
    a = prepare_tensor(photo(glyph, dark_on_light=True), robust=robust)
    b = prepare_tensor(photo(glyph, dark_on_light=False), robust=robust)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("robust", [False, True])
@pytest.mark.parametrize("size", [120, 280, 600])
def test_digit_is_centred_by_centre_of_mass(mnist_samples, photo, size, robust):
    for _, glyph in mnist_samples:
        image = prepare_tensor(photo(glyph, size=size), robust=robust)[0, :, :, 0]
        ys, xs = np.indices(image.shape)
        cy, cx = (ys * image).sum() / image.sum(), (xs * image).sum() / image.sum()
        assert abs(cy - 13.5) < 0.8 and abs(cx - 13.5) < 0.8


def test_digit_fits_20px_box(mnist_samples, photo):
    for _, glyph in mnist_samples:
        image = prepare_tensor(photo(glyph))[0, :, :, 0] > 0.05
        rows, cols = np.flatnonzero(image.any(axis=1)), np.flatnonzero(image.any(axis=0))
        assert np.ptp(rows) + 1 <= 24 and np.ptp(cols) + 1 <= 24  # 20px box + resampling/roll slack


def test_stretch_contrast_is_opt_in(mnist_samples):
    """Default keeps raw stroke intensity; the flag rescales the stroke peak to 1.0."""
    _, glyph = mnist_samples[0]
    faint = (glyph.astype(np.float32) * 0.4).astype(np.uint8)  # light-on-dark, peak ~100
    default = prepare_from_gray(faint)
    stretched = prepare_from_gray(faint, stretch_contrast=True)
    assert default.max() < 0.6
    assert stretched.max() > 0.9


def test_otsu_separates_bimodal_image():
    gray = np.concatenate([np.full(500, 20, np.uint8), np.full(500, 220, np.uint8)])
    assert 20 <= otsu_threshold(gray) < 220


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("robust", [False, True])
def test_blank_image_is_rejected(robust):
    with pytest.raises(PreprocessError) as info:
        prepare_tensor(_png(np.full((64, 64), 255, np.uint8)), robust=robust)
    assert info.value.status_code == 422


@pytest.mark.parametrize("payload", [b"", b"definitely not an image"])
def test_unreadable_bytes_are_rejected(payload):
    with pytest.raises(PreprocessError) as info:
        prepare_tensor(payload)
    assert info.value.status_code == 415


def test_pixel_limit_is_enforced():
    with pytest.raises(PreprocessError) as info:
        prepare_tensor(_png(np.zeros((100, 100), np.uint8)), max_pixels=50)
    assert info.value.status_code == 413


def _degrade(png_bytes: bytes, seed: int) -> bytes:
    """Photo with uneven lighting plus sensor noise: the two failure modes found by eval/pipeline_eval.py."""
    gray = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("L"), dtype=np.float32)
    ys, xs = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]] / gray.shape[0]
    gray = gray * (1.0 - 0.4 * (xs + ys) / 2) + np.random.default_rng(seed).normal(0, 18, gray.shape)
    return _png(np.clip(gray, 0, 255).astype(np.uint8))


def test_robust_mode_survives_shadow_and_noise(client, mnist_samples, photo):
    """Regression guard for the pipeline fix. Fixed seeds; >= 9/10 allows for numeric drift across libs."""
    correct = 0
    for index, (digit, glyph) in enumerate(mnist_samples):
        data = _degrade(photo(glyph), seed=index)
        tensor = prepare_tensor(data, robust=True, stretch_contrast=True)
        correct += int(client.app.state.classifier.model.predict(tensor, verbose=0).argmax() == digit)
    assert correct >= 9
