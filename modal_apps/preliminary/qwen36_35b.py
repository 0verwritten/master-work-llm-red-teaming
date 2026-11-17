from __future__ import annotations

from preliminary_study.modal_app import build_app
from preliminary_study.roster import QWEN36_35B_A3B_VISION

app, Server = build_app(QWEN36_35B_A3B_VISION)
