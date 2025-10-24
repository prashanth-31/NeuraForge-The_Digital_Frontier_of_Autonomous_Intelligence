FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files first (for caching)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi --only main --no-root

# Copy application code
COPY . .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

# Run database migrations before starting the API
CMD ["./docker-entrypoint.sh"]
