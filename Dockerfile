# syntax=docker/dockerfile:1.7
FROM python:3.9-slim

WORKDIR /app

ENV HF_HOME=/app/hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=true \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# cache-friendly: only invalidates when requirements.txt changes
COPY requirements.txt .

# One install layer: upgrade pip, install uv, then install deps with uv
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip uv \
    && uv pip install --system -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu



# Minimal sanity check
RUN python -c "import transformers, torch; print('transformers', transformers.__version__, 'torch', torch.__version__)"

# App code last (so code changes don’t bust deps layer)
COPY . .

RUN mkdir -p cache results \
    && useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8501

# Production flags (no live reload)
CMD ["streamlit", "run", "batch_inference_final.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false","--server.fileWatcherType=none"]
