import React, { useRef, useState, useEffect, useCallback } from "react";
import { motion } from "framer-motion";

export default function DrawingPad({ onRecognize, loading }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const stateRef = useRef({
    strokes: [],
    activeStroke: null,
    strokeWidth: 20,
    timer: null,
    isDrawing: false,
  });

  const [strokeWidth, setStrokeWidth] = useState(20);
  const [hasStrokes, setHasStrokes] = useState(false);
  const [realtime, setRealtime] = useState(true);

  // Resize canvas to match display container with HiDPI support
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const size = Math.floor(Math.min(rect.width, rect.height || rect.width));
    if (size <= 0) return;

    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;

    redraw();
  }, []);

  useEffect(() => {
    resizeCanvas();
    const ro = new ResizeObserver(() => resizeCanvas());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [resizeCanvas]);

  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const size = canvas.width / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    // Subtle guide grid inside drawing area
    ctx.strokeStyle = "rgba(184, 201, 166, 0.4)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 6]);
    ctx.strokeRect(size * 0.15, size * 0.15, size * 0.7, size * 0.7);
    ctx.setLineDash([]);

    const { strokes, activeStroke } = stateRef.current;
    const allStrokes = activeStroke ? [...strokes, activeStroke] : strokes;

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#213AA0";

    for (const stroke of allStrokes) {
      if (!stroke.points || stroke.points.length === 0) continue;
      ctx.lineWidth = stroke.width || 20;

      if (stroke.points.length === 1) {
        const pt = stroke.points[0];
        ctx.fillStyle = "#213AA0";
        ctx.beginPath();
        ctx.arc(pt.x * size, pt.y * size, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }

      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x * size, stroke.points[0].y * size);

      for (let i = 1; i < stroke.points.length - 1; i++) {
        const p1 = stroke.points[i];
        const p2 = stroke.points[i + 1];
        const midX = ((p1.x + p2.x) / 2) * size;
        const midY = ((p1.y + p2.y) / 2) * size;
        ctx.quadraticCurveTo(p1.x * size, p1.y * size, midX, midY);
      }

      const last = stroke.points[stroke.points.length - 1];
      ctx.lineTo(last.x * size, last.y * size);
      ctx.stroke();
    }
  }, []);

  const triggerRecognition = useCallback(() => {
    if (stateRef.current.strokes.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Export cleanly to standard 280x280 canvas with high contrast for model
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = 280;
    exportCanvas.height = 280;
    const ctx = exportCanvas.getContext("2d");

    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, 280, 280);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";

    for (const stroke of stateRef.current.strokes) {
      if (!stroke.points || stroke.points.length === 0) continue;
      ctx.lineWidth = ((stroke.width || 20) / (canvas.width / (window.devicePixelRatio || 1))) * 280;

      if (stroke.points.length === 1) {
        const pt = stroke.points[0];
        ctx.fillStyle = "#000000";
        ctx.beginPath();
        ctx.arc(pt.x * 280, pt.y * 280, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }

      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x * 280, stroke.points[0].y * 280);
      for (let i = 1; i < stroke.points.length - 1; i++) {
        const p1 = stroke.points[i];
        const p2 = stroke.points[i + 1];
        ctx.quadraticCurveTo(
          p1.x * 280,
          p1.y * 280,
          ((p1.x + p2.x) / 2) * 280,
          ((p1.y + p2.y) / 2) * 280
        );
      }
      const last = stroke.points[stroke.points.length - 1];
      ctx.lineTo(last.x * 280, last.y * 280);
      ctx.stroke();
    }

    onRecognize(exportCanvas, "Drawing");
  }, [onRecognize]);

  const getCanvasPoint = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX ?? (e.touches && e.touches[0]?.clientX) ?? 0;
    const clientY = e.clientY ?? (e.touches && e.touches[0]?.clientY) ?? 0;
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    return { x, y };
  };

  const handlePointerDown = (e) => {
    e.preventDefault();
    canvasRef.current?.setPointerCapture?.(e.pointerId);
    clearTimeout(stateRef.current.timer);

    const pt = getCanvasPoint(e);
    stateRef.current.isDrawing = true;
    stateRef.current.activeStroke = {
      points: [pt],
      width: strokeWidth,
    };
    redraw();
  };

  const handlePointerMove = (e) => {
    if (!stateRef.current.isDrawing || !stateRef.current.activeStroke) return;
    const pt = getCanvasPoint(e);
    stateRef.current.activeStroke.points.push(pt);
    redraw();
  };

  const handlePointerUp = () => {
    if (!stateRef.current.isDrawing) return;
    stateRef.current.isDrawing = false;
    if (stateRef.current.activeStroke && stateRef.current.activeStroke.points.length > 0) {
      stateRef.current.strokes.push(stateRef.current.activeStroke);
      stateRef.current.activeStroke = null;
      setHasStrokes(true);
      redraw();

      if (realtime) {
        clearTimeout(stateRef.current.timer);
        stateRef.current.timer = setTimeout(() => {
          triggerRecognition();
        }, 380);
      }
    }
  };

  const handleUndo = () => {
    if (stateRef.current.strokes.length === 0) return;
    stateRef.current.strokes.pop();
    setHasStrokes(stateRef.current.strokes.length > 0);
    redraw();
    if (stateRef.current.strokes.length > 0) {
      triggerRecognition();
    }
  };

  const handleClear = () => {
    clearTimeout(stateRef.current.timer);
    stateRef.current.strokes = [];
    stateRef.current.activeStroke = null;
    setHasStrokes(false);
    redraw();
  };

  // Quick preset sample generator (0-9) to test immediately
  const handleQuickSample = (digit) => {
    const canvas = document.createElement("canvas");
    canvas.width = 280;
    canvas.height = 280;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, 280, 280);
    ctx.font = 'bold 180px "Space Grotesk", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#000000";
    ctx.fillText(String(digit), 140, 140);
    onRecognize(canvas, `Preset "${digit}"`);
  };

  return (
    <div className="flex flex-col gap-3">
      {/* Canvas container */}
      <div
        ref={containerRef}
        className="relative w-full aspect-square rounded-lg border-2 border-ink bg-[#D3DEC2] overflow-hidden shadow-inner flex items-center justify-center select-none"
      >
        <canvas
          ref={canvasRef}
          className="drawing-canvas touch-none block"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />

        {!hasStrokes && (
          <div className="absolute inset-0 pointer-events-none flex flex-col items-center justify-center gap-1.5 text-center px-6">
            <span className="font-display font-semibold text-lg text-ink">Draw a digit here</span>
            <span className="text-muted text-xs">Write with mouse, stylus, or finger</span>
            <div className="mt-2 flex items-center gap-1.5 text-[11px] text-muted/80 bg-paper/60 px-2.5 py-1 rounded-full border border-rule/50">
              <span>Auto-reads on pen lift</span>
            </div>
          </div>
        )}

        {loading && (
          <div className="absolute top-3 right-3 flex items-center gap-2 bg-ink/90 text-paper text-xs font-medium px-2.5 py-1 rounded-md shadow-md backdrop-blur-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-stamp-glow animate-ping" />
            <span>Analyzing…</span>
          </div>
        )}
      </div>

      {/* Control buttons & stroke width */}
      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleUndo}
            disabled={!hasStrokes || loading}
            className="px-3 py-1.5 text-xs font-semibold rounded border border-ink/40 text-ink bg-transparent hover:bg-ink/10 active:scale-95 disabled:opacity-40 disabled:pointer-events-none transition-all"
            title="Undo stroke (Ctrl+Z)"
          >
            Undo
          </button>
          <button
            type="button"
            onClick={handleClear}
            disabled={!hasStrokes || loading}
            className="px-3 py-1.5 text-xs font-semibold rounded border border-ink/40 text-ink bg-transparent hover:bg-ink/10 active:scale-95 disabled:opacity-40 disabled:pointer-events-none transition-all"
            title="Clear canvas"
          >
            Clear
          </button>
          <button
            type="button"
            onClick={triggerRecognition}
            disabled={!hasStrokes || loading}
            className="px-4 py-1.5 text-xs font-bold rounded bg-ink text-paper hover:bg-ink-deep active:scale-95 disabled:opacity-40 disabled:pointer-events-none transition-all shadow-sm"
          >
            Run Now
          </button>
        </div>

        {/* Thickness selector */}
        <div className="flex items-center gap-2 bg-paper-dim/60 px-2 py-1 rounded border border-rule/60 text-xs text-muted">
          <span>Width:</span>
          {[14, 20, 28].map((w) => (
            <button
              key={w}
              type="button"
              onClick={() => setStrokeWidth(w)}
              className={`w-6 h-6 rounded flex items-center justify-center transition-all ${
                strokeWidth === w ? "bg-ink text-paper font-bold" : "hover:bg-paper"
              }`}
            >
              <span
                className="rounded-full bg-current block"
                style={{ width: `${w / 3.5}px`, height: `${w / 3.5}px` }}
              />
            </button>
          ))}
        </div>
      </div>

      {/* Quick Test Digits */}
      <div className="mt-1 pt-2 border-t border-rule/40">
        <div className="flex items-center justify-between text-[11px] text-muted mb-1.5">
          <span>Quick test digits:</span>
          <span className="text-[10px] opacity-75">Click to run immediately</span>
        </div>
        <div className="grid grid-cols-10 gap-1">
          {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((d) => (
            <button
              key={d}
              type="button"
              onClick={() => handleQuickSample(d)}
              className="py-1 rounded border border-rule hover:border-ink hover:bg-ink/10 font-display font-bold text-xs text-ink text-center transition-all active:scale-90"
              title={`Test digit ${d}`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
