# app/routers/history.py
from fastapi import APIRouter, Query
from ..models.schemas import HistoryItem, HistoryResponse
from ..services.storage import list_history
from datetime import datetime

router = APIRouter(prefix="/api", tags=["history"])

@router.get("/history", response_model=HistoryResponse)
def history(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):
    total, rows = list_history(offset, limit)
    items = [
        HistoryItem(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["ts"]),
            filename=row.get("filename", ""),
            predicted_class=row["predicted_class"],
            topk=row["topk"]
        )
        for row in rows
    ]
    return HistoryResponse(total=total, items=items)
