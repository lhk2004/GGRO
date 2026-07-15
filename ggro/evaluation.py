"""Lightweight local metrics for GGRO generation outputs."""

from __future__ import annotations

import re
from typing import Optional


def extract_multiple_choice_answer(text: str) -> Optional[str]:
    match = re.search(r"(?i)answer\s*[:\-]?\s*([A-J])\b", text)
    return match.group(1).upper() if match else None
