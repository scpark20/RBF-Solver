# Compose builds from this directory. GPU drivers remain a host prerequisite.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/RBF-Solver
COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r /tmp/requirements.lock
# Source and assets are bind mounted by compose; no private data is baked into an image.
CMD ["python","sit/dashboard.py","--host","0.0.0.0"]
