from __future__ import annotations
from typing import Optional
from preprocessing import detect_lang_heuristic
from utils import ensure_lang_code
from config import SUPPORTED_LANGUAGES, DEFAULT_LANG


def select_language(user_text: str, requested_lang: Optional[str] = None) -> str:
    lang = ensure_lang_code(requested_lang, default=DEFAULT_LANG)
    if lang in SUPPORTED_LANGUAGES:
        return lang
    # Kembali ke heuristic jika tidak diketahui
    guessed = detect_lang_heuristic(user_text)
    if guessed in SUPPORTED_LANGUAGES:
        return guessed
    return DEFAULT_LANG
