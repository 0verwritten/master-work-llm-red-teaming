from __future__ import annotations

from preliminary_study.modal_app import build_app
from preliminary_study.roster import GPT_OSS_20B_TEXT_ONLY

app, Server = build_app(GPT_OSS_20B_TEXT_ONLY)
