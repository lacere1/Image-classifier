# Build context is the REPOSITORY ROOT (see docker-compose.yml), because the
# service imports the `landmarklens` package and loads a trained checkpoint
# from artifacts/ -- neither of which lives under backend/.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch: the full CUDA wheel is ~2.5 GB and useless in this image.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /app/backend/requirements.txt

COPY landmarklens /app/landmarklens
COPY backend /app/backend
COPY artifacts /app/artifacts

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    LANDMARKLENS_CHECKPOINT=/app/artifacts/best_model.pt \
    LANDMARKLENS_THREADS=4

EXPOSE 8000

# One worker, several threads: the model is loaded once per worker and holds a
# few hundred MB, and Grad-CAM is serialised behind a lock inside the process.
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 4 -t 120 -b 0.0.0.0:${PORT} --chdir /app/backend app:app"]
