from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import predict, metrics, history

app = FastAPI(title="Image Classification API",
    description="API para clasificar imágenes usando un modelo TensorFlow entrenado.",
    version="1.0.0",
    contact={
        "name": "Paulo Cesar Albis - Miguel Angel Osorio",
        "url": "https://github.com/PauloCesarAlvis",
    },)

# Ajusta ORIGINS al dominio/puerto del frontend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://miguelosorio11.github.io",   # si lo sirves en GH Pages u otro host
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(metrics.router)
app.include_router(history.router)

@app.get("/api/health")
def health():
    return {"status": "ok"}
