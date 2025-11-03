# ================================
# Etapa base: Python + TensorFlow
# ================================
FROM python:3.12-slim

# Evitar buffering de logs
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Copiar requirements primero (para aprovechar cache)
COPY requirements.txt .

# Instalar dependencias del sistema requeridas por TensorFlow y PIL
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
        && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Exponer el puerto de FastAPI
EXPOSE 8000

# Comando por defecto: lanzar el servidor
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
