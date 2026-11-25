import React, { useRef, useState, useEffect, useCallback } from "react";

export default function CameraView({ onRecognize, loading }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraState, setCameraState] = useState("idle"); // 'idle' | 'starting' | 'active' | 'captured' | 'error'
  const [errorMessage, setErrorMessage] = useState("");
  const [facingMode, setFacingMode] = useState("environment"); // 'environment' | 'user'
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [flash, setFlash] = useState(false);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraState("idle");
  }, []);

  const startCamera = useCallback(async () => {
    stopCamera();
    setErrorMessage("");
    setCapturedImage(null);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setErrorMessage("Camera access is not supported on this browser or origin (requires HTTPS or localhost).");
      setCameraState("error");
      return;
    }

    setCameraState("starting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facingMode },
          width: { ideal: 1280 },
          height: { ideal: 1280 },
        },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setCameraState("active");

      // Check device count for flip button
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const videoInputs = devices.filter((d) => d.kind === "videoinput");
        setHasMultipleCameras(videoInputs.length > 1);
      }).catch(() => {});
    } catch (err) {
      let msg = "Could not start camera.";
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
        msg = "Camera permission was denied. Please allow camera access in your browser settings.";
      } else if (err.name === "NotFoundError") {
        msg = "No camera found on this device.";
      } else if (err.name === "NotReadableError") {
        msg = "Camera is in use by another application.";
      }
      setErrorMessage(msg);
      setCameraState("error");
    }
  }, [facingMode, stopCamera]);

  useEffect(() => {
    return () => stopCamera();
  }, [stopCamera]);

  const capturePhoto = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    // Trigger flash animation
    setFlash(true);
    setTimeout(() => setFlash(false), 300);

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const GUIDE = 0.7; // Middle 70% frame
    const side = Math.round(Math.min(vw, vh) * GUIDE);
    const sx = Math.round((vw - side) / 2);
    const sy = Math.round((vh - side) / 2);
    const out = Math.min(side, 640);

    const canvas = document.createElement("canvas");
    canvas.width = out;
    canvas.height = out;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, sx, sy, side, side, 0, 0, out, out);

    setCapturedImage(canvas.toDataURL("image/jpeg", 0.9));
    setCameraState("captured");
    stopCamera();

    onRecognize(canvas, "Camera");
  };

  const toggleCameraFacing = () => {
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
  };

  useEffect(() => {
    if (cameraState === "active" || cameraState === "starting") {
      startCamera();
    }
  }, [facingMode]);

  return (
    <div className="flex flex-col gap-3">
      {/* Video / Preview Surface */}
      <div className="relative w-full aspect-square rounded-lg border-2 border-ink bg-[#D3DEC2] overflow-hidden shadow-inner flex items-center justify-center">
        {/* Flash animation overlay */}
        {flash && <div className="absolute inset-0 bg-white z-20 pointer-events-none animate-pulse" />}

        {/* Live video */}
        <video
          ref={videoRef}
          playsInline
          muted
          autoPlay
          className={`w-full h-full object-cover ${cameraState === "active" ? "block" : "hidden"}`}
        />

        {/* Captured preview */}
        {cameraState === "captured" && capturedImage && (
          <img
            src={capturedImage}
            alt="Captured digit"
            className="w-full h-full object-contain bg-black/80"
          />
        )}

        {/* Viewfinder overlay when camera is active */}
        {cameraState === "active" && (
          <div className="absolute inset-[15%] pointer-events-none z-10">
            {/* Corner brackets */}
            <div className="absolute top-0 left-0 w-6 h-6 border-t-2 border-l-2 border-stamp-glow" />
            <div className="absolute top-0 right-0 w-6 h-6 border-t-2 border-r-2 border-stamp-glow" />
            <div className="absolute bottom-0 left-0 w-6 h-6 border-b-2 border-l-2 border-stamp-glow" />
            <div className="absolute bottom-0 right-0 w-6 h-6 border-b-2 border-r-2 border-stamp-glow" />
            <div className="absolute inset-0 border border-white/30 rounded-sm" />
          </div>
        )}

        {/* Idle prompt */}
        {cameraState === "idle" && (
          <div className="flex flex-col items-center justify-center text-center p-6 gap-3">
            <div className="w-16 h-16 rounded-full bg-paper/70 border border-rule flex items-center justify-center text-ink">
              <svg viewBox="0 0 24 24" className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                <circle cx="12" cy="13" r="4" />
              </svg>
            </div>
            <span className="font-display font-semibold text-lg text-ink">Position digit in frame</span>
            <span className="text-muted text-xs max-w-xs">Use dark pen on plain paper with good, even lighting</span>
            <button
              type="button"
              onClick={startCamera}
              className="mt-2 px-5 py-2 rounded bg-ink text-paper font-semibold text-sm hover:bg-ink-deep active:scale-95 transition-all shadow-sm"
            >
              Open Camera
            </button>
          </div>
        )}

        {/* Starting indicator */}
        {cameraState === "starting" && (
          <div className="flex flex-col items-center gap-2 text-ink">
            <span className="w-6 h-6 border-2 border-ink border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-semibold">Starting camera…</span>
          </div>
        )}

        {/* Error message */}
        {cameraState === "error" && (
          <div className="p-6 text-center text-stamp-deep flex flex-col items-center gap-2">
            <span className="font-display font-bold text-base">Camera Error</span>
            <span className="text-xs max-w-xs">{errorMessage}</span>
            <button
              type="button"
              onClick={startCamera}
              className="mt-3 px-4 py-1.5 text-xs font-semibold rounded border border-stamp-deep text-stamp-deep hover:bg-stamp/10"
            >
              Try Again
            </button>
          </div>
        )}
      </div>

      {/* Camera Action Buttons */}
      <div className="flex items-center justify-between gap-2 pt-1">
        {cameraState === "active" && (
          <>
            <button
              type="button"
              onClick={capturePhoto}
              disabled={loading}
              className="flex-1 py-2.5 px-4 rounded bg-stamp text-white font-bold text-sm hover:bg-stamp-deep active:scale-95 transition-all flex items-center justify-center gap-2 shadow-sm"
            >
              <svg viewBox="0 0 24 24" className="w-4 h-4 fill-current">
                <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2" />
                <circle cx="12" cy="12" r="5" />
              </svg>
              <span>Capture Digit</span>
            </button>
            {hasMultipleCameras && (
              <button
                type="button"
                onClick={toggleCameraFacing}
                className="px-3 py-2 text-xs font-semibold rounded border border-ink/40 text-ink hover:bg-ink/10"
                title="Flip camera"
              >
                Flip
              </button>
            )}
            <button
              type="button"
              onClick={stopCamera}
              className="px-3 py-2 text-xs font-semibold rounded border border-ink/40 text-ink hover:bg-ink/10"
            >
              Cancel
            </button>
          </>
        )}

        {cameraState === "captured" && (
          <div className="flex items-center gap-2 w-full">
            <button
              type="button"
              onClick={startCamera}
              className="flex-1 py-2 px-4 rounded bg-ink text-paper font-semibold text-xs hover:bg-ink-deep active:scale-95 transition-all"
            >
              Retake Photo
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
