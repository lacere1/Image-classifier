FROM python:3.11-slim

# Install minimal system deps; libgl is used by some Pillow backends
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libjpeg62-turbo && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Because the build context is ./backend, these files are at the root of the context
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the backend sources
COPY . /app

EXPOSE 8000
# Gunicorn with 2 workers and threads for concurrency
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8000", "app:app"]