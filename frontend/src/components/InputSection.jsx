import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import DrawingPad from "./DrawingPad";
import CameraView from "./CameraView";
import UploadView from "./UploadView";

const TABS = [
  { id: "draw", label: "Draw", icon: (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 19l7-7 3 3-7 7-3-3z" />
      <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z" />
      <path d="M2 2l7.586 7.586" />
      <circle cx="11" cy="11" r="2" />
    </svg>
  )},
  { id: "camera", label: "Camera", icon: (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
      <circle cx="12" cy="13" r="4" />
    </svg>
  )},
  { id: "upload", label: "Upload", icon: (
    <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  )},
];

export default function InputSection({ mode, onModeChange, onRecognize, loading }) {
  return (
    <div className="sheet paper-bg rounded-lg p-5 shadow-card border border-ink/20">
      {/* Header with Mode Tabs */}
      <div className="flex items-center justify-between gap-2 mb-4 pb-2 border-b border-rule">
        <h2 className="font-display font-bold text-xl text-ink tracking-tight">Input Digit</h2>

        {/* Tab switch bar */}
        <div className="flex p-0.5 rounded-md bg-[#D3DEC2] border border-rule/70 text-xs">
          {TABS.map((tab) => {
            const active = mode === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onModeChange(tab.id)}
                className={`relative px-3 py-1.5 font-semibold rounded-[4px] transition-all flex items-center gap-1.5 ${
                  active ? "text-paper shadow-sm" : "text-muted hover:text-ink"
                }`}
              >
                {active && (
                  <motion.div
                    layoutId="activeTabBadge"
                    className="absolute inset-0 bg-ink rounded-[4px]"
                    transition={{ type: "spring", stiffness: 450, damping: 35 }}
                  />
                )}
                <span className="relative z-10">{tab.icon}</span>
                <span className="relative z-10">{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Mode content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.2 }}
        >
          {mode === "draw" && (
            <DrawingPad onRecognize={onRecognize} loading={loading} />
          )}
          {mode === "camera" && (
            <CameraView onRecognize={onRecognize} loading={loading} />
          )}
          {mode === "upload" && (
            <UploadView onRecognize={onRecognize} loading={loading} />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
