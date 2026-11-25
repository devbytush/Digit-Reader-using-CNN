// API layer for the Digit Reader React frontend

const API_URL = import.meta.env.VITE_API_URL || "";

/** POST an image file/blob to /v1/convert and return the parsed response. */
export async function convertImage(blob) {
  const form = new FormData();
  form.append("file", blob, blob.type === "image/png" ? "digit.png" : "digit.jpg");
  const res = await fetch(`${API_URL}/v1/convert`, { method: "POST", body: form });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(typeof data.detail === "string" ? data.detail : "Recognition failed");
    err.status = res.status;
    throw err;
  }
  return data;
}

/** POST a base64 image string to /v1/convert-base64. */
export async function convertBase64(imageData) {
  const res = await fetch(`${API_URL}/v1/convert-base64`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageData }),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(typeof data.detail === "string" ? data.detail : "Recognition failed");
    err.status = res.status;
    throw err;
  }
  return data;
}

/** Liveness check: returns true if the API is ready. */
export async function checkHealth() {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 12000);
    const res = await fetch(`${API_URL}/ready`, { signal: ctrl.signal });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

/** Server and model metrics. */
export async function getMetrics() {
  const res = await fetch(`${API_URL}/v1/metrics`);
  if (!res.ok) throw new Error("Failed to fetch metrics");
  return res.json();
}

/** Open a WebSocket for real-time drawing recognition. */
export function createDrawSocket(onResult, onError) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/draw`);
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.error) onError?.(data.error);
      else onResult?.(data);
    } catch { /* ignore malformed frames */ }
  };
  ws.onerror = () => onError?.("WebSocket connection error");
  return ws;
}

/** Convert a canvas to a Blob. */
export function canvasToBlob(canvas, type = "image/png", quality = 0.92) {
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error("Could not encode the image"))),
      type,
      quality,
    ),
  );
}
