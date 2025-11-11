"""
Preprocessing teks dan parsing dataset.
Mendukung format input fleksibel dan normalisasi yang sadar bahasa.
"""

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Tuple, Iterable, Optional
from pathlib import Path
from utils import log_info, log_warn, log_error
from config import SUPPORTED_LANGUAGES, DEFAULT_LANG

# Normalisasi dasar: simpan huruf, angka, tanda baca; padatkan spasi.
_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Normalisasi Unicode
    text = unicodedata.normalize("NFKC", text)
    # Hapus URL (opsional)
    text = _URL_RE.sub(" ", text)
    # Padatkan spasi
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def detect_lang_heuristic(text: str) -> str:
    """
    Heuristik sangat ringan untuk menebak bahasa ketika hilang:
    - Jika mengandung banyak Hiragana/Katakana/Kanji → JP
    - Jika banyak kata umum Indonesia → ID
    - Selain itu default EN
    """
    if not text:
        return DEFAULT_LANG

    jp_chars = sum(1 for ch in text if "\u3040" <= ch <= "\u30ff" or "\u4e00" <= ch <= "\u9faf")
    if jp_chars >= max(2, len(text) // 10):
        return "JP"

    # Petunjuk Indonesia yang sangat naif
    id_markers = ["yang", "dan", "di", "untuk", "dengan", "tidak", "akan", "itu", "ini", "apa", "bagaimana"]
    hit = sum(1 for w in id_markers if f" {w} " in f" {text.lower()} ")
    if hit >= 2:
        return "ID"

    return "EN"


def parse_data_file(path: Path) -> List[Dict]:
    """
    Terima format fleksibel:
    - Baris JSONL: {"lang":"EN","input":"...","response":"..."}
    - TSV: lang<TAB>input<TAB>response
    - Mirip CSV dengan 3 kolom (lang,input,response) - toleran terhadap koma dalam teks jika dikutip
    Jika 'lang' hilang, coba deteksi heuristik.
    """
    if not path.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {path}")

    rows: List[Dict] = []
    # Coba JSONL terlebih dahulu
    try:
        import json
        with path.open("r", encoding="utf-8") as f:
            is_jsonl = True
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    lang = obj.get("lang") or obj.get("language")
                    text_in = obj.get("input") or obj.get("prompt") or obj.get("question")
                    text_out = obj.get("response") or obj.get("answer")
                    if not text_in or not text_out:
                        raise ValueError("Input/response hilang")
                    lang = (lang or detect_lang_heuristic(text_in)).upper()
                    rows.append(
                        {
                            "lang": lang if lang in SUPPORTED_LANGUAGES else detect_lang_heuristic(text_in),
                            "input": normalize_text(text_in),
                            "response": normalize_text(text_out),
                        }
                    )
                except Exception:
                    is_jsonl = False
                    break
        if is_jsonl and rows:
            return rows
        rows.clear()
    except Exception:
        rows.clear()

    # Coba TSV
    try:
        with path.open("r", encoding="utf-8") as f:
            tsv_like = True
            for i, line in enumerate(f, 1):
                parts = [p.strip() for p in line.strip().split("\t")]
                if len(parts) < 2:
                    tsv_like = False
                    break
                if len(parts) == 2:
                    lang = detect_lang_heuristic(parts[0])
                    text_in, text_out = parts[0], parts[1]
                else:
                    lang, text_in, text_out = parts[0].upper(), parts[1], parts[2]
                rows.append(
                    {
                        "lang": lang if lang in SUPPORTED_LANGUAGES else detect_lang_heuristic(text_in),
                        "input": normalize_text(text_in),
                        "response": normalize_text(text_out),
                    }
                )
        if tsv_like and rows:
            return rows
        rows.clear()
    except Exception:
        rows.clear()

    # Fallback: CSV
    try:
        import csv
        with path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for i, parts in enumerate(reader, 1):
                if not parts:
                    continue
                if len(parts) >= 3:
                    lang, text_in, text_out = parts[0].strip().upper(), parts[1], parts[2]
                elif len(parts) == 2:
                    text_in, text_out = parts[0], parts[1]
                    lang = detect_lang_heuristic(text_in)
                else:
                    continue
                rows.append(
                    {
                        "lang": lang if lang in SUPPORTED_LANGUAGES else detect_lang_heuristic(text_in),
                        "input": normalize_text(text_in),
                        "response": normalize_text(text_out),
                    }
                )
        return rows
    except Exception as e:
        log_error("Gagal mem-parse dataset", error=str(e))
        raise