import { useState, useEffect, useRef, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import gsap from "gsap";
import Lenis from "lenis";
import Header from "./components/Header";
import InputSection from "./components/InputSection";
import ResultSection from "./components/ResultSection";
import History from "./components/History";
import { convertImage, checkHealth } from "./api";

const DIGIT_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine"];

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [mode, setMode] = useState("draw");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const mainRef = useRef(null);

  // ── Lenis smooth scroll ──
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced) return;
    const lenis = new Lenis({ duration: 1.1, smoothWheel: true });
    const raf = (time) => { lenis.raf(time); requestAnimationFrame(raf); };
    requestAnimationFrame(raf);
    return () => lenis.destroy();
  }, []);

  // ── GSAP entrance animation ──
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from("[data-animate]", {
        y: 28, opacity: 0, duration: 0.75, stagger: 0.12,
        ease: "power3.out", delay: 0.15,
      });
    }, mainRef);
    return () => ctx.revert();
  }, []);

  // ── API health check ──
  useEffect(() => {
    checkHealth().then((ok) => setApiStatus(ok ? "connected" : "offline"));
  }, []);

  // ── Recognition handler ──
  const handleRecognize = useCallback(async (canvas, source) => {
    setLoading(true);
    setError(null);
    try {
      const blob = await new Promise((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error("Could not encode image"))),
          source === "Drawing" ? "image/png" : "image/jpeg",
          0.92,
        ),
      );
      const t0 = performance.now();
      const data = await convertImage(blob);
      const ms = Math.round(performance.now() - t0);

      const reading = {
        id: Date.now(),
        digit: data.digit,
        text: data.text,
        word: data.word || DIGIT_WORDS[data.digit] || String(data.digit),
        confidence: data.confidence,
        top_predictions: data.top_predictions,
        warnings: data.warnings || [],
        source,
        ms,
      };
      setResult(reading);
      setHistory((prev) => [reading, ...prev].slice(0, 12));
      setApiStatus("connected");
    } catch (err) {
      setError(err.message);
      if (err.status >= 500 || !err.status) setApiStatus("offline");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div ref={mainRef} className="min-h-screen bg-desk text-on-desk font-body">
      {/* ── Ambient background ── */}
      <div className="fixed inset-0 pointer-events-none z-0" aria-hidden="true">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_15%_0%,rgba(33,58,160,0.08)_0%,transparent_55%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_85%_100%,rgba(213,57,31,0.05)_0%,transparent_55%)]" />
        <div className="absolute inset-0 ambient-grid" />
      </div>

      <div className="relative z-10">
        <Header
          apiStatus={apiStatus}
          onRecheck={() => {
            setApiStatus("checking");
            checkHealth().then((ok) => setApiStatus(ok ? "connected" : "offline"));
          }}
        />

        <main className="max-w-[1120px] mx-auto px-4 sm:px-6 lg:px-8 pb-24">
          {/* ── Title ── */}
          <div className="pt-24 sm:pt-28 pb-8" data-animate>
            <h1 className="font-display text-4xl sm:text-5xl lg:text-[3.6rem] font-bold tracking-tight leading-[0.9]">
              <span className="bg-gradient-to-br from-on-desk via-on-desk to-on-desk-muted bg-clip-text text-transparent">
                Digit Reader
              </span>
            </h1>
            <p className="mt-4 text-on-desk-muted max-w-lg text-[15px] sm:text-base leading-relaxed">
              Point a camera at a handwritten digit, drop a photo, or draw one — the CNN reads it instantly.
            </p>
          </div>

          {/* ── Main panels ── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
            <div data-animate>
              <InputSection
                mode={mode}
                onModeChange={setMode}
                onRecognize={handleRecognize}
                loading={loading}
              />
            </div>
            <div data-animate>
              <ResultSection
                result={result}
                loading={loading}
                error={error}
              />
            </div>
          </div>

          {/* ── History ── */}
          <AnimatePresence>
            {history.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                transition={{ duration: 0.4, ease: [0.3, 0.8, 0.2, 1] }}
              >
                <History readings={history} current={result} onSelect={setResult} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* ── Keyboard hints ── */}
          <footer className="hidden md:flex flex-wrap gap-x-5 gap-y-2 mt-10 text-on-desk-muted text-[13px]" data-animate>
            {[
              ["C", "Camera"], ["U", "Upload"], ["D", "Draw"],
              ["Space", "Capture"], ["Esc", "Close camera"], ["Ctrl+Z", "Undo stroke"],
            ].map(([key, label]) => (
              <span key={key} className="flex items-center gap-2">
                <kbd className="min-w-[22px] px-1.5 py-0.5 border border-white/20 border-b-2 rounded text-center text-xs font-semibold text-on-desk">
                  {key}
                </kbd>
                {label}
              </span>
            ))}
          </footer>
        </main>
      </div>
    </div>
  );
}
