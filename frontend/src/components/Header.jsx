import { motion } from "framer-motion";
import clsx from "clsx";

const STATUS = {
  checking: { dot: "bg-on-desk-muted animate-pulse-soft", label: "Checking API…" },
  connected: { dot: "bg-green-400", label: "Connected" },
  offline: { dot: "bg-stamp-glow", label: "API offline" },
};

export default function Header({ apiStatus, onRecheck }) {
  const s = STATUS[apiStatus] || STATUS.checking;

  return (
    <header className="fixed top-0 left-0 right-0 z-30 pointer-events-none">
      <div className="max-w-[1120px] mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between gap-3">
        {/* Logo */}
        <motion.a
          href="/"
          className="pointer-events-auto flex items-center gap-2.5 px-3 py-2 rounded-md bg-desk/80 backdrop-blur-md"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
          </svg>
          <span className="font-display text-lg font-bold tracking-wide">Digit Reader</span>
        </motion.a>

        {/* Right side */}
        <div className="pointer-events-auto flex items-center gap-3">
          {/* Visualizer link */}
          <a
            href="/visualize"
            className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-md border border-white/15 bg-desk/70 backdrop-blur-md text-[13px] font-semibold text-on-desk-muted hover:text-on-desk hover:border-white/30 transition-colors"
          >
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 3 3 8l9 5 9-5-9-5Z" />
              <path d="m3 13 9 5 9-5" />
            </svg>
            Visualize
          </a>

          {/* API status */}
          <button
            onClick={onRecheck}
            className={clsx(
              "flex items-center gap-2 px-3 py-2 rounded-md border bg-desk/70 backdrop-blur-md text-[13px] transition-colors",
              apiStatus === "connected"
                ? "border-white/15 text-on-desk-muted hover:border-white/30"
                : "border-stamp-glow/30 text-on-desk-muted hover:border-stamp-glow/50",
            )}
          >
            <span className={clsx("w-2 h-2 rounded-full flex-none", s.dot)} />
            <span className="hidden sm:inline">{s.label}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
