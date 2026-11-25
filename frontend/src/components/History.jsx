import React from "react";
import { motion } from "framer-motion";

export default function History({ readings, current, onSelect }) {
  if (!readings || readings.length === 0) return null;

  return (
    <div className="mt-8 pt-6 border-t border-white/10">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-display font-bold text-sm tracking-wide text-on-desk-muted uppercase">
          Earlier Readings in this Session
        </h3>
        <span className="text-xs text-on-desk-muted font-mono">
          {readings.length} {readings.length === 1 ? "entry" : "entries"}
        </span>
      </div>

      <div className="flex gap-2.5 overflow-x-auto pb-2 history-scroll">
        {readings.map((reading) => {
          const isSelected = current && current.id === reading.id;
          return (
            <motion.button
              key={reading.id}
              type="button"
              onClick={() => onSelect(reading)}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.96 }}
              className={`flex-none flex items-center gap-3 p-2.5 pr-4 rounded-md border transition-all text-left ${
                isSelected
                  ? "bg-paper border-stamp shadow-md"
                  : "bg-paper/90 border-transparent hover:border-white/30"
              }`}
            >
              {/* Digit Box */}
              <div className="w-10 h-10 rounded bg-[#D3DEC2] flex items-center justify-center font-display font-extrabold text-2xl text-ink border border-ink/30">
                {reading.digit}
              </div>

              {/* Meta */}
              <div className="flex flex-col">
                <span className="text-xs font-bold text-ink-deep font-display leading-tight">
                  {reading.word || `Digit ${reading.digit}`}
                </span>
                <span className="text-[11px] font-mono text-muted">
                  {(reading.confidence * 100).toFixed(1)}% · {reading.source}
                </span>
              </div>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
