from fastapi import APIRouter, UploadFile, File, HTTPException
from ..services.model import ModelService
from ..services.storage import insert_history
from ..models.schemas import PredictResponse

router = APIRouter(prefix="/api", tags=["predict"])

@router.post("/predict", summary="Clasificar imagen con modelo TensorFlow",
    description="Recibe un archivo de imagen y devuelve la clase predicha y las probabilidades top-k.",
  response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        pred, conf, topk = ModelService.instance().predict(image_bytes, top_k=3)
        insert_history(image.filename, pred, topk)
        return PredictResponse(
            predicted_class=pred,
            confidence=conf,
            topk=topk,
            filename=image.filename
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
