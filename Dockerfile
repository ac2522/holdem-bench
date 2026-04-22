# TODO: pin digest before v0.1.0 release
FROM python:3.12-slim

# Install uv via official installer
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev

COPY evals ./evals
COPY tests ./tests

ENTRYPOINT ["uv", "run", "holdembench"]
