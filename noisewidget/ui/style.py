"""Colours shared by every pane, matching MicroscopeVuilder's workspace."""

from __future__ import annotations

from ..physics.base import Status

BACKGROUND = "#0b0d10"
PANEL = "#14171c"
GRID = "#262b33"
AXIS = "#7a828e"
TEXT = "#c9d1d9"
MUTED = "#9aa3b0"

STATUS_COLORS = {
    Status.PASS: "#4fbf78",
    Status.WARN: "#d9a441",
    Status.FAIL: "#d95f5f",
    Status.INFO: "#7a828e",
}

# One colour per noise term, used identically in the graph, the budget bar and the
# toggles, so "orange" means read noise everywhere.
TERM_COLORS = {
    # Validated as a categorical set on the #0b0d10 surface (CVD and normal-vision
    # separation, >= 3:1 contrast). Order matters: it is the validated adjacency.
    "photon_shot": "#3987e5",
    "read": "#d95926",
    "prnu": "#199e70",
    "dark_shot": "#c98500",
    "background_shot": "#d55181",
    "cic": "#9085e9",
    "quantization": "#7a828e",  # minor term: drawn dashed, never colour alone
    "total": "#e6edf3",
    "signal": "#e6edf3",
}

BANNER = "background: #3a2f18; color: #d9c45a; padding: 4px; border-radius: 3px;"
