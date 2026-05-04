FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY for_fl ./for_fl

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

CMD ["python", "-c", "import for_fl.server, for_fl.client; print('Flower app image is ready')"]
