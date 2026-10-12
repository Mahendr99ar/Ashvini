FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
COPY ashvini ./ashvini
COPY models ./models
RUN pip install ".[model]"
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn ashvini.api.main:app --host 0.0.0.0 --port ${PORT}"]
