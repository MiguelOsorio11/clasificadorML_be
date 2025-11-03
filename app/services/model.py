from pathlib import Path
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow import keras

ARTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTS_DIR / "best_model.keras"
CLASSES_PATH = ARTS_DIR / "class_names.json"

IMG_SIZE = 224

class ModelService:
    _instance = None

    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH, compile=False)
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _preprocess(self, image_bytes: bytes) -> np.ndarray:
        from io import BytesIO
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img).astype("float32")
        arr = preprocess_input(arr)        # igual que en entrenamiento
        x = np.expand_dims(arr, 0)
        return x

    def predict(self, image_bytes: bytes, top_k: int = 3):
        x = self._preprocess(image_bytes)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred = self.class_names[idx]
        # top-k ordenado
        order = np.argsort(probs)[::-1][:top_k]
        topk = {self.class_names[i]: float(probs[i]) for i in order}
        return pred, float(probs[idx]), topk
