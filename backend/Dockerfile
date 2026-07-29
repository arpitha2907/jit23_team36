FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-asm \
    tesseract-ocr-ben \
    tesseract-ocr-guj \
    tesseract-ocr-hin \
    tesseract-ocr-kan \
    tesseract-ocr-mal \
    tesseract-ocr-mar \
    tesseract-ocr-ori \
    tesseract-ocr-pan \
    tesseract-ocr-san \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    tesseract-ocr-urd \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install the CPU-only PyTorch build first (a few hundred MB instead of the
# ~2GB+ default CUDA build) — this app only ever runs the model on CPU.
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV UPLOAD_DIR=/data/uploads
RUN mkdir -p /data/uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
