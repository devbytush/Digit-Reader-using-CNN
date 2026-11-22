# Handwritten Digit Recognizer


https://digit-reader-using-cnn.onrender.com
A FastAPI app that accepts a photo of one handwritten digit, preprocesses it for
the included MNIST CNN, and returns the predicted digit with class probabilities.
The browser page supports both taking a camera photo and choosing an image from
the gallery.

## Run locally

Use Python 3.14 or another Python version supported by TensorFlow, then install
and start the app:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. The JSON endpoint is `POST /predict` with an image
in a multipart form field named `file`; interactive API documentation is at
`/docs`.

The app includes FastAPI's CORS middleware for separate frontends. By default,
it allows development UIs on ports 3000 and 5173. For deployment, set
`UI_ORIGINS` to a comma-separated list of exact frontend origins, for example:

```powershell
$env:UI_ORIGINS = "https://digits.example.com"
python -m uvicorn main:app
```

For a phone camera to work, open the app over HTTPS or from `localhost` on the
phone itself. The API accepts images up to 10 MB and 20 megapixels. This model
recognizes one digit per image; use a clear, high-contrast photo for best results.
