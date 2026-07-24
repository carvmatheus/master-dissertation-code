"""Persistência auditável das decisões do MCKP em JSONL."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict


_LOCK = Lock()


def append_audit(path: str, record: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
