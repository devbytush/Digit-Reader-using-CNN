import React, { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";

export default function ResultSection({ result, loading, error }) {
  const [copied, setCopied] = useState(false);
  const [peekGrid, setPeekGrid] = useState(false);
  const [hoverPixel, setHoverPixel] = useState(null); // { x, y, value }
  const gridCanvasRef = useRef(null);

  const handleCopy = () => {
    if (!result) return;
    navigator.clipboard.writeText(result.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    });
  };

  const confidence = result ? result.confidence : 0;
  const isLowConf = confidence < 0.7;

  return (
    <div className="sheet paper-bg rounded-lg p-5 shadow-card border border-ink/20 flex flex-col gap-4">
      {/* Plate Header */}
      <div className="flex items-center justify-between pb-2 border-b border-rule">
        <h2 className="font-display font-bold text-xl text-ink tracking-tight">Model Reading</h2>
        {result && (
          <span className="text-xs font-mono text-muted bg-paper-dim/80 px-2 py-0.5 rounded border border-rule/60">
            {result.source} · {result.ms}ms
          </span>
        )}
      </div>

      {/* Reading Plate (28x28 grid + Stamped Digit) */}
      <div className="relative w-full aspect-square rounded-lg border-2 border-ink bg-[#D3DEC2] overflow-hidden shadow-inner flex items-center justify-center select-none group">
        {/* Empty State */}
        {!result && !loading && !error && (
          <div className="flex flex-col items-center justify-center p-6 text-center text-muted gap-2">
            <div className="w-16 h-16 rounded-full border border-rule flex items-center justify-center bg-paper/50">
              <span className="font-display font-bold text-2xl text-ink opacity-40">?</span>
            </div>
            <span className="font-display font-semibold text-lg text-ink">No reading yet</span>
            <span className="text-xs max-w-xs">Draw, upload, or capture a handwritten digit to see the model output</span>
          </div>
        )}

        {/* Loading Sweep */}
        {loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-paper-dim/80 backdrop-blur-sm z-30">
            <div className="w-12 h-12 rounded-full border-3 border-ink border-t-transparent animate-spin mb-3" />
            <span className="font-display font-bold text-sm text-ink">Reading Neural Features…</span>
            <span className="text-xs text-muted font-mono mt-1">Passing through Conv2D layers</span>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="p-6 text-center text-stamp-deep flex flex-col items-center gap-2 z-20">
            <div className="w-12 h-12 rounded-full bg-stamp/10 flex items-center justify-center text-stamp-deep mb-1">
              <svg viewBox="0 0 24 24" className="w-6 h-6" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <span className="font-display font-bold text-base">Recognition Failed</span>
            <span className="text-xs max-w-xs font-medium">{error}</span>
          </div>
        )}

        {/* Reading Result */}
        {result && !loading && (
          <>
            {/* Background Simulated 28x28 Grid Texture */}
            <div
              className={`absolute inset-0 p-4 transition-opacity duration-300 ${
                peekGrid ? "opacity-100" : "opacity-30"
              }`}
            >
              <div className="w-full h-full grid grid-cols-28 gap-px bg-rule/30 p-1 rounded">
                {/* 28x28 = 784 cells */}
                {Array.from({ length: 784 }).map((_, i) => (
                  <div
                    key={i}
                    className="aspect-square bg-ink/10 rounded-none transition-colors"
                  />
                ))}
              </div>
            </div>

            {/* Giant Ink Stamp */}
            <div
              className={`relative z-10 flex flex-col items-center justify-center pointer-events-none transition-all duration-300 ${
                peekGrid ? "opacity-10 scale-95" : "opacity-100 scale-100"
              }`}
            >
              <motion.div
                initial={{ scale: 1.8, rotate: -15, opacity: 0 }}
                animate={{ scale: 1, rotate: -3, opacity: 1 }}
                transition={{ type: "spring", stiffness: 350, damping: 25 }}
                className="relative"
              >
                {/* Stamp Shadow / Ghost */}
                <span className="absolute inset-0 font-display font-extrabold text-[160px] sm:text-[200px] leading-none text-stamp/30 translate-x-1.5 translate-y-1 blur-[1px]">
                  {result.digit}
                </span>
                {/* Stamp Foreground */}
                <span className="relative font-display font-extrabold text-[160px] sm:text-[200px] leading-none text-stamp">
                  {result.digit}
                </span>
              </motion.div>
            </div>

            {/* Hold to Peek Hint */}
            <div
              onPointerDown={() => setPeekGrid(true)}
              onPointerUp={() => setPeekGrid(false)}
              onPointerLeave={() => setPeekGrid(false)}
              className="absolute bottom-2.5 right-2.5 z-20 cursor-pointer bg-paper/90 border border-ink/30 px-2.5 py-1 rounded text-[11px] font-semibold text-ink shadow-sm hover:bg-paper active:scale-95 transition-all"
            >
              {peekGrid ? "Release to stamp" : "Hold to peek grid"}
            </div>
          </>
        )}
      </div>

      {/* Metrics & Details Block */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-ink">
        {/* Main Recognized Result */}
        <div className="p-3 bg-paper rounded border border-ink/20 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-muted uppercase tracking-wider">Reading</span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="font-display font-extrabold text-3xl text-ink">
              {result ? result.digit : "–"}
            </span>
            <span className="text-xs font-semibold text-muted capitalize">
              {result ? result.word : ""}
            </span>
          </div>
          <button
            type="button"
            onClick={handleCopy}
            disabled={!result}
            className="mt-2 text-[11px] font-semibold py-1 px-2 rounded border border-ink/30 hover:bg-ink/10 active:scale-95 disabled:opacity-40 transition-all text-center"
          >
            {copied ? "Copied!" : "Copy Text"}
          </button>
        </div>

        {/* Confidence Gauge */}
        <div className="p-3 bg-paper rounded border border-ink/20 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-muted uppercase tracking-wider">Confidence</span>
          <div className="mt-1">
            <span className={`font-display font-bold text-2xl ${isLowConf ? "text-stamp-deep" : "text-ink"}`}>
              {result ? `${(confidence * 100).toFixed(1)}%` : "–"}
            </span>
            {/* Confidence Bar */}
            <div className="w-full h-2 bg-paper-dim rounded-full mt-2 overflow-hidden border border-rule">
              <div
                className={`h-full transition-all duration-700 ease-out rounded-full ${
                  isLowConf ? "bg-stamp" : "bg-ink"
                }`}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
          </div>
          <span className="text-[10px] text-muted mt-2">
            {isLowConf ? "Low confidence warning" : "High confidence match"}
          </span>
        </div>

        {/* Runners-up / Top 3 Candidates */}
        <div className="col-span-2 sm:col-span-1 p-3 bg-paper rounded border border-ink/20 flex flex-col justify-between">
          <span className="text-[11px] font-semibold text-muted uppercase tracking-wider">Runners-Up</span>
          <div className="flex flex-col gap-1 mt-1">
            {result && result.top_predictions ? (
              result.top_predictions.slice(1, 3).map((cand) => (
                <div key={cand.digit} className="flex items-center justify-between text-xs">
                  <span className="font-display font-bold text-ink">#{cand.digit}</span>
                  <span className="font-mono text-muted text-[11px]">
                    {(cand.probability * 100).toFixed(1)}%
                  </span>
                </div>
              ))
            ) : (
              <span className="text-xs text-muted font-mono">–</span>
            )}
          </div>
          <span className="text-[10px] text-muted mt-2">Alternative predictions</span>
        </div>
      </div>

      {/* Warnings Banner if any */}
      {result && result.warnings && result.warnings.length > 0 && (
        <div className="p-2.5 rounded bg-stamp/10 border border-stamp/30 text-stamp-deep text-xs flex items-center gap-2">
          <svg viewBox="0 0 24 24" className="w-4 h-4 flex-none" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>{result.warnings[0]}</span>
        </div>
      )}
    </div>
  );
}
