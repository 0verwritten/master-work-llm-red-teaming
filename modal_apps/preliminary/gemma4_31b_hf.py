from __future__ import annotations

from preliminary_study.modal_hf_app import build_hf_app
from preliminary_study.roster import GEMMA4_31B_VISION_HF

app, Server = build_hf_app(GEMMA4_31B_VISION_HF)
