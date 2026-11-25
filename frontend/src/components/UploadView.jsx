import React, { useRef, useState, useEffect } from "react";

export default function UploadView({ onRecognize, loading }) {
  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  const processFile = (file) => {
    if (!file) return;
    setErrorMsg("");

    if (!file.type.startsWith("image/")) {
      setErrorMsg("Please upload an image file (PNG, JPG, WebP).");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setErrorMsg("Image exceeds maximum 10MB limit.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        // Downscale to max 1280px to keep request snappy
        const maxDim = 1280;
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));

        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#FFFFFF";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        setPreviewUrl(canvas.toDataURL("image/jpeg", 0.9));
        onRecognize(canvas, "Upload");
      };
      img.onerror = () => setErrorMsg("Could not decode image.");
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  // Clipboard paste support
  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            processFile(file);
            break;
          }
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  return (
    <div className="flex flex-col gap-3">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            processFile(e.target.files[0]);
            e.target.value = "";
          }
        }}
      />

      {/* Drop Surface */}
      <div
        onClick={() => fileInputRef.current?.click()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative w-full aspect-square rounded-lg border-2 border-dashed transition-all cursor-pointer overflow-hidden flex flex-col items-center justify-center p-6 text-center select-none ${
          isDragging
            ? "border-stamp bg-stamp/10 scale-[0.99]"
            : "border-ink/50 bg-[#D3DEC2] hover:border-ink hover:bg-paper-dim/80"
        }`}
      >
        {previewUrl ? (
          <img
            src={previewUrl}
            alt="Uploaded digit"
            className="w-full h-full object-contain p-2"
          />
        ) : (
          <div className="flex flex-col items-center gap-2 text-ink">
            <div className="w-14 h-14 rounded-full bg-paper/80 border border-rule flex items-center justify-center shadow-sm">
              <svg viewBox="0 0 24 24" className="w-7 h-7 text-ink" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <span className="font-display font-semibold text-lg">Drop photo here</span>
            <span className="text-muted text-xs">or click to browse from device</span>
            <span className="text-[11px] text-muted/80 bg-paper/60 px-2 py-0.5 rounded border border-rule/50 mt-1">
              Supports clipboard paste (Ctrl+V)
            </span>
          </div>
        )}

        {loading && (
          <div className="absolute inset-0 bg-paper/60 backdrop-blur-sm flex flex-col items-center justify-center gap-2">
            <span className="w-7 h-7 border-3 border-ink border-t-transparent rounded-full animate-spin" />
            <span className="text-xs font-semibold text-ink">Processing digit…</span>
          </div>
        )}
      </div>

      {errorMsg && (
        <span className="text-xs font-medium text-stamp-deep text-center">{errorMsg}</span>
      )}

      {/* Button */}
      <div className="flex items-center justify-between gap-2 pt-1">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="w-full py-2 px-4 rounded bg-ink text-paper font-semibold text-xs hover:bg-ink-deep active:scale-95 transition-all shadow-sm"
        >
          {previewUrl ? "Choose Another Photo" : "Choose a Photo"}
        </button>
      </div>
    </div>
  );
}
