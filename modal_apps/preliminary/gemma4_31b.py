from __future__ import annotations

from preliminary_study.modal_app import build_app
from preliminary_study.roster import GEMMA4_31B_TEXT_ONLY

app, Server = build_app(GEMMA4_31B_TEXT_ONLY)
