"""Re-apply the P0 wording fix to GUI tooltip + unified_bayesian.py comment.

Binary-mode replacement to preserve CRLF line endings exactly.
"""
from pathlib import Path

# Edit 1: GUI tooltip
gui = Path("spectral_predict_gui_optimized.py")
data = gui.read_bytes()
old = (
    b'        _persist_tooltip = (\r\n'
    b'            "Auto (default): first 10 trials in-memory; SQLite enabled only "\r\n'
    b'            "when median trial fit time > 1s (overhead ~1.2x at threshold). "\r\n'
    b'            "Fast models like PLS run 8x slower with SQLite enabled, so auto "\r\n'
    b'            "disables it for them.\\n"\r\n'
    b'            "Always on: SQLite from trial 0 \xe2\x80\x94 universal crash-resume at the "\r\n'
    b'            "cost of speed (PLS ~8x slower, XGBoost ~1.4x slower). Auto-set "\r\n'
    b'            "for the session when you accept the resume banner so the loaded "\r\n'
    b'            "SQLite is actually used.\\n"\r\n'
    b'            "Always off: pure in-memory, zero overhead. Use when interactive "\r\n'
    b'            "speed matters and re-run on crash is cheap."\r\n'
    b'        )\r\n'
)
new = (
    b'        _persist_tooltip = (\r\n'
    b'            "Auto (default): first 10 trials in-memory; SQLite enabled only "\r\n'
    b'            "when median trial fit time > 1s. Auto keeps SQLite off for "\r\n'
    b'            "fast models (sub-1s trials) where per-trial write cost would "\r\n'
    b'            "be the largest fraction of trial time.\\n"\r\n'
    b'            "Always on: SQLite from trial 0 \xe2\x80\x94 universal crash-resume with "\r\n'
    b'            "near-zero overhead (all models within ~1.06x of in-memory in "\r\n'
    b'            "benchmarks). Auto-set for the session when you accept the "\r\n'
    b'            "resume banner so the loaded SQLite is actually used.\\n"\r\n'
    b'            "Always off: pure in-memory."\r\n'
    b'        )\r\n'
)
count = data.count(old)
assert count == 1, f"GUI tooltip: expected 1 occurrence, got {count}"
gui.write_bytes(data.replace(old, new))
print("GUI tooltip updated")

# Edit 2: unified_bayesian.py threshold comment
ub = Path("src/spectral_predict/unified_bayesian.py")
data2 = ub.read_bytes()
old2 = b"    _AUTO_THRESHOLD_S = 1.0  # median fit > 1.0s -> SQLite ON (ratio ~1.2x)\r\n"
new2 = b"    _AUTO_THRESHOLD_S = 1.0  # median fit > 1.0s -> SQLite ON (post-T-42 ratio ~1.0-1.06x)\r\n"
count2 = data2.count(old2)
assert count2 == 1, f"unified_bayesian comment: expected 1 occurrence, got {count2}"
ub.write_bytes(data2.replace(old2, new2))
print("unified_bayesian comment updated")
