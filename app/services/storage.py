# app/services/storage.py
from __future__ import annotations
import json, os, tempfile, threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Archivo de historial a nivel de proyecto
HISTORY_PATH = Path(__file__).resolve().parents[2] / "history.json"

_lock = threading.Lock()

def _ensure():
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text(json.dumps({"total": 0, "items": []}, ensure_ascii=False, indent=2), encoding="utf-8")

def _atomic_write_text(path: Path, data: str):
    """Escritura atómica para evitar archivos corruptos."""
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="hist_", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp_path, path)  # atomic swap
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass

_ensure()

def insert_history(filename: str, predicted_class: str, topk: Dict[str, float]) -> int:
    """Agrega una entrada al JSON y devuelve el id asignado."""
    with _lock:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        items: List[Dict] = data.get("items", [])
        new_id = (items[-1]["id"] + 1) if items else 1
        row = {
            "id": new_id,
            "ts": datetime.utcnow().isoformat(),
            "filename": filename or "",
            "predicted_class": predicted_class,
            "topk": topk
        }
        items.append(row)
        data["items"] = items
        data["total"] = len(items)
        _atomic_write_text(HISTORY_PATH, json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        return new_id

def list_history(offset: int, limit: int) -> Tuple[int, List[Dict]]:
    """Devuelve (total, slice) ordenado DESC por id."""
    data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    items: List[Dict] = data.get("items", [])
    # Orden descendente por id
    items_desc = items[::-1]
    slice_ = items_desc[offset:offset+limit]
    return len(items), slice_
