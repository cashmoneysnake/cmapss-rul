FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch (the default Linux wheel pulls in GPU libraries and is several GB)
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY artifacts ./artifacts

ENV MODEL_DIR=/app/artifacts/fd004_gru
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
