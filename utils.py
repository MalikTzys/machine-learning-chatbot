from __future__ import annotations
import time
import json
from contextlib import contextmanager
from typing import Any, Dict


def log_info(msg: str, **kwargs):
    print(json.dumps({"level": "INFO", "msg": msg, **kwargs}))


def log_warn(msg: str, **kwargs):
    print(json.dumps({"level": "WARN", "msg": msg, **kwargs}))


def log_error(msg: str, **kwargs):
    print(json.dumps({"level": "ERROR", "msg": msg, **kwargs}))


@contextmanager
def timed(msg: str):
    start = time.time()
    log_info(f"{msg} - start")
    try:
        yield
    finally:
        dur = time.time() - start
        log_info(f"{msg} - done", duration_ms=int(dur * 1000))


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def ensure_lang_code(lang: str | None, default: str = "EN") -> str:
    if not lang:
        return default
    return lang.strip().upper()
