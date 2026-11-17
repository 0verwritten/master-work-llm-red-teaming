from __future__ import annotations

import re
from typing import Literal

LanguageAdherence = Literal["pass", "fail", "indeterminate"]

ENGLISH = "en"
UKRAINIAN = "uk"


def detect_language_adherence(text: str, expected: str) -> LanguageAdherence:

    latin = len(re.findall(r"[A-Za-z]", text))
    cyrillic = len(re.findall(r"[Ѐ-ӿ]", text))
    ukrainian_specific = len(re.findall(r"[іїєґІЇЄҐ]", text))
    if latin + cyrillic < 3:
        return "indeterminate"
    if expected == ENGLISH:
        if cyrillic > latin:
            return "fail"
        return "pass" if latin >= 3 else "indeterminate"
    if latin > cyrillic:
        return "fail"
    if cyrillic >= 3 and (ukrainian_specific > 0 or latin == 0):
        return "pass"
    return "indeterminate"
