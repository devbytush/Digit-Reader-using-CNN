/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        desk:      { DEFAULT: "#1C2A26", light: "#243530", 2: "#141E1B" },
        paper:     { DEFAULT: "#E3EBD6", dim: "#D3DEC2" },
        ink:       { DEFAULT: "#213AA0", deep: "#16257A", glow: "#6F8CFF" },
        stamp:     { DEFAULT: "#D5391F", deep: "#A82A15", glow: "#FF6A4D" },
        muted:     "#55634F",
        "on-desk": { DEFAULT: "#DCE6CF", muted: "#8FA597" },
        rule:      { DEFAULT: "#B8C9A6", soft: "#D0DCC2", dim: "#9DB38B" },
        amber:     "#7A5200",
      },
      fontFamily: {
        display: ['"Space Grotesk"', "system-ui", "sans-serif"],
        body:    ["Inter", "system-ui", "sans-serif"],
        mono:    ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "8px",
      },
      boxShadow: {
        card:     "0 4px 24px rgba(0,0,0,0.12), 0 1px 4px rgba(0,0,0,0.08)",
        "card-lg":"0 8px 40px rgba(0,0,0,0.18), 0 2px 8px rgba(0,0,0,0.1)",
        glow:     "0 0 20px rgba(111,140,255,0.15)",
        stamp:    "0 0 24px rgba(213,57,31,0.25)",
      },
      animation: {
        "fade-in":    "fadeIn 0.5s ease-out forwards",
        "slide-up":   "slideUp 0.6s cubic-bezier(0.16,1,0.3,1) forwards",
        "pulse-soft": "pulseSoft 2s ease-in-out infinite",
        "stamp-land": "stampLand 0.52s cubic-bezier(0.2,0.9,0.25,1.08) forwards",
        "grid-sweep": "gridSweep 1.2s ease-out forwards",
      },
      keyframes: {
        fadeIn:     { from: { opacity: "0" },                                    to: { opacity: "1" } },
        slideUp:    { from: { opacity: "0", transform: "translateY(20px)" },     to: { opacity: "1", transform: "translateY(0)" } },
        pulseSoft:  { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.5" } },
        stampLand:  { "0%": { opacity: "0", transform: "rotate(-10deg) scale(1.6)" }, "55%": { opacity: "1", transform: "rotate(-3deg) scale(0.97)" }, "100%": { opacity: "1", transform: "rotate(-3deg) scale(1)" } },
        gridSweep:  { from: { clipPath: "inset(0 100% 0 0)" },                  to: { clipPath: "inset(0 0 0 0)" } },
      },
    },
  },
  plugins: [],
};
