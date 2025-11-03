from fastapi import APIRouter, HTTPException
from ..models.schemas import MetricsResponse
from pathlib import Path
import json

router = APIRouter(prefix="/api", tags=["metrics"])
ARTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"

@router.get("/metrics", summary="Obtener métricas del modelo",
    description="Devuelve las métricas del modelo entrenado (accuracy, matriz de confusión, etc.)",
  response_model=MetricsResponse)
def get_metrics():
    p = ARTS_DIR / "metrics.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="metrics.json no encontrado")
    data = json.loads(p.read_text(encoding="utf-8"))
    return MetricsResponse(
        test_loss=data["test_loss"],
        test_accuracy=data["test_accuracy"],
        report=data["report"],
        confusion_matrix=data["confusion_matrix"]
    )
