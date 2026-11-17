from __future__ import annotations

from preliminary_study.modal_app import build_app
from preliminary_study.roster import MISTRAL_SMALL_32_VISION

app, Server = build_app(MISTRAL_SMALL_32_VISION)
