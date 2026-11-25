<div align="center">

<img src="assets/banner.svg" alt="Digit Reader: handwritten digit to text" width="100%">

<br>

[![Live demo](https://img.shields.io/badge/Live%20demo-Open%20on%20Render-1C2A26?style=for-the-badge&logo=render&logoColor=white)](https://digit-reader-using-cnn.onrender.com)

![Python](https://img.shields.io/badge/Python-3.10%2B-213AA0?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-D5391F?style=flat-square&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-1C2A26?style=flat-square&logo=fastapi&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-55634F?style=flat-square)

**Point a camera at a handwritten digit, or draw one, and get it back as text.**

[Live demo](https://digit-reader-using-cnn.onrender.com) · [How it works](#how-it-works) · [Run locally](#run-it-locally) · [API](#api-reference) · [Deploy](#deploy-on-render)

</div>

---

## Overview

**Digit Reader** is an end-to-end handwritten digit recognizer. A convolutional neural network trained on the MNIST dataset sits behind a FastAPI service, and a single-file web app lets you feed it digits three ways: live camera, photo upload, or drawing on the page.

It is built to cope with real photos rather than only clean MNIST test images. Photos are cleaned up, cropped and centered before the model sees them.

> **Try it now:** [digit-reader-using-cnn.onrender.com](https://digit-reader-using-cnn.onrender.com)
> The demo runs on Render's free tier, so the first request after a quiet period can take 30 to 60 seconds while the server wakes up.

## Features

| | |
|---|---|
| **Three ways in** | Open the camera and capture, upload or drop a photo, paste an image, or draw with a mouse or finger. |
| **Live drawing** | Draw mode reads automatically half a second after you lift the pen. |
| **See what the model sees** | The result panel shows the 28 × 28 grid the digit was reduced to. Hover or press and hold it to peek behind the stamp. |
| **Confidence and runners-up** | Every reading comes with a confidence score and the next two most likely digits. Low confidence is flagged. |
| **Robust preprocessing** | Auto-inversion, Otsu thresholding, noise rejection, cropping and center-of-mass centering. |
| **Clean JSON API** | Typed responses, size limits, health and readiness probes, and interactive docs at `/docs`. |
| **Copy and history** | Copy the result as text and revisit earlier readings in the session. |
| **3D network visualizer** | A scroll-driven tour of every layer, from the input pixels to the final decision, plus training curves, a confusion matrix and the filters the first layer learned. Open it from the **Visualize** button. |
| **Keyboard friendly** | `C` camera, `U` upload, `D` draw, `Space` capture, `Esc` close camera, `Ctrl+Z` undo a stroke. |

## How it works

<img src="assets/how-it-works.svg" alt="Pipeline: capture, clean up, shrink to 28 by 28, read" width="100%">

1. **Capture.** The browser sends one image to `POST /v1/convert`. The camera crops to the on-screen frame and uploads are downscaled first to keep requests small.
2. **Clean up.** The API fixes EXIF rotation, flattens transparency onto white, converts to grayscale and inverts dark-on-light writing so the digit is bright on a dark background, as in MNIST. It then applies an Otsu threshold and keeps the largest block of ink, which drops stray specks and shadows.
3. **Shrink to 28 × 28.** The digit is cropped to its bounding box, scaled to fit 20 px, placed on a 28 × 28 canvas and centered by its center of mass. This matches how MNIST digits were prepared.
4. **Read.** The CNN outputs a probability for each digit 0 to 9. The API returns the best guess, its spelled-out word, the confidence and the top three candidates.

## Tech stack

- **Model:** TensorFlow / Keras CNN trained on MNIST, exported as `mnist_cnn.keras`
- **Backend:** FastAPI, Uvicorn, Pydantic v2, NumPy, Pillow
- **Frontend:** one dependency-free `index.html` (vanilla JavaScript, canvas, `getUserMedia`)
- **Hosting:** Render

## Project structure

```text
Digit-Reader-using-CNN/
├── main.py                       # FastAPI app: preprocessing, inference, endpoints
├── index.html                    # Frontend served at "/" (camera, upload, draw)
├── visualize.html                # 3D network visualizer served at "/visualize"
├── visualize.py                  # Builds exported_model/viz_data.json for the visualizer
├── requirements.txt              # Python dependencies
├── exported_model/
│   ├── mnist_cnn.keras           # Trained CNN
│   └── viz_data.json             # Training history and test results (made by visualize.py)
├── MNIST_CNN_Digit_Recognition (1).ipynb      # Model training notebook
├── CNN_Digit_Recognition_Project_Report (2).pdf   # Project report
└── assets/                       # README images
```

## Run it locally

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone
git clone https://github.com/devbytush/Digit-Reader-using-CNN.git
cd Digit-Reader-using-CNN

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload
```

Open **http://localhost:8000**. The app and the API are served from the same address, and `localhost` counts as a secure origin, so the browser will offer camera access.

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

> **Seeing TensorFlow log lines on startup?** Messages about `cpu_feature_guard` or `absl::InitializeLog()` are informational and harmless. They only say TensorFlow is using your CPU's AVX2/FMA instructions.

### Testing the camera from a phone

Browsers only allow camera access on `localhost` or HTTPS. To test on a phone, expose port 8000 through an HTTPS tunnel such as ngrok or cloudflared and open the tunnel URL. The page and API then share one origin, so no extra setup is needed.

## Network visualizer

Press **Visualize** on the main page to see how the network reads your digit. Scroll to follow the signal through every layer in 3D, then see how the model was trained. Drag the scene to look around, hover a feature map for details, and use the **Try another digit** panel to draw or upload a new one and watch every layer update.

The 3D tour works out of the box. To add the training curves, test accuracy, confusion matrix and sample digits, generate the data once:

```bash
# in your training notebook, save the history
#   import json; json.dump(history.history, open("history.json", "w"))

python visualize.py --history history.json
```

This loads `exported_model/mnist_cnn.keras`, evaluates it on the MNIST test set and writes `exported_model/viz_data.json`. Commit that file to include the training section on your hosted demo. Offline? Pass a local copy of the dataset with `--mnist-npz mnist.npz`.

## API reference

### `POST /v1/convert`

Send one image as `multipart/form-data` in a field named `file`.

```bash
curl -X POST -F "file=@digit.jpg" https://digit-reader-using-cnn.onrender.com/v1/convert
```

**Response**

```json
{
  "text": "7",
  "digit": 7,
  "word": "seven",
  "confidence": 0.9931,
  "top_predictions": [
    { "digit": 7, "probability": 0.9931 },
    { "digit": 1, "probability": 0.0049 },
    { "digit": 2, "probability": 0.0011 }
  ],
  "warnings": []
}
```

`warnings` contains a message when confidence is below 70%.

**Errors**

| Status | Meaning |
|---|---|
| `400` | The uploaded file is empty |
| `413` | Image is over 10 MB or 20 megapixels |
| `415` | Not a supported or readable image |
| `422` | No digit found, or it is too small or faint |
| `503` | The model is not loaded |

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /` | The web app |
| `GET /health` | Liveness probe |
| `GET /ready` | Readiness probe, succeeds once the model is loaded |
| `GET /visualize` | The 3D network visualizer |
| `GET /v1/model` | Architecture, kernels and training data for the visualizer |
| `POST /v1/trace` | Runs an image and returns every layer's activations |
| `GET /docs` | Interactive Swagger UI |

### Configuration

Settings can be overridden with environment variables prefixed `D2T_`:

| Variable | Default | Description |
|---|---|---|
| `D2T_MODEL_PATH` | `exported_model/mnist_cnn.keras` | Path to the Keras model |
| `D2T_CORS_ORIGINS` | localhost dev origins | JSON list of allowed origins, for example `["https://my-site.com"]` |
| `D2T_MAX_UPLOAD_BYTES` | `10485760` | Maximum upload size |
| `D2T_MAX_IMAGE_PIXELS` | `20000000` | Maximum image size in pixels |
| `D2T_LOW_CONFIDENCE_THRESHOLD` | `0.70` | Below this, a warning is returned |

## Deploy on Render

1. Create a new **Web Service** from this repository.
2. Set the build command to `pip install -r requirements.txt`.
3. Set the start command to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
4. Deploy. Render gives you a public HTTPS URL, which also satisfies the browser's camera requirement.

Because the frontend is served by the same app, it automatically calls the API on its own origin.

## Tips for the best results

- Write **one digit** in dark ink on plain paper.
- Fill the dashed frame with the digit and keep the lighting even, without shadows.
- If confidence is low, retake the photo closer and straighter, or draw the digit instead.

## Limitations

- The model reads **one digit at a time**, 0 to 9. Multi-digit numbers are not supported.
- It was trained on MNIST, so unusual styles, very thin strokes or heavy background texture can reduce accuracy.
- On the free Render tier the first request after idle time is slow.

## Roadmap

- [ ] Multi-digit numbers through digit segmentation
- [ ] Letter and symbol recognition (EMNIST)
- [ ] Test-time augmentation for tougher photos
- [ ] Dockerfile for one-command deployment

## Documentation

- Model training: [`MNIST_CNN_Digit_Recognition (1).ipynb`](<./MNIST_CNN_Digit_Recognition (1).ipynb>)
- Project report: [`CNN_Digit_Recognition_Project_Report (2).pdf`](<./CNN_Digit_Recognition_Project_Report (2).pdf>)

## Author

Built by **[@devbytush](https://github.com/devbytush)**. If this project helped you, consider giving it a star.

<div align="center">

<sub>Digit Reader · CNN · FastAPI · Render</sub>

</div>
