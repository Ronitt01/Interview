# Demo image. CPU-only on purpose: this serves inference, and the whole point of
# section 11 is that the quantised model runs in single-digit milliseconds on a CPU.
FROM python:3.13-slim

# libsndfile is soundfile's native dependency; ffmpeg lets librosa read anything
# a reviewer might upload to the demo.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements before source, so a code edit does not invalidate the pip layer.
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY demo/ ./demo/
COPY data/hinglish/ ./data/hinglish/
COPY weights/ ./weights/
COPY README.md report/ ./

ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    OMP_NUM_THREADS=4

EXPOSE 7860

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["python", "demo/app.py", "--host", "0.0.0.0", "--port", "7860"]
