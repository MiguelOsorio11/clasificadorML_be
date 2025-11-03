from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    topk: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    filename: Optional[str] = None

class MetricsResponse(BaseModel):
    test_loss: float
    test_accuracy: float
    report: dict
    confusion_matrix: List[List[int]]

class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    filename: str
    predicted_class: str
    topk: Dict[str, float]

class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]
