# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim


RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# Application specific setup
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/


WORKDIR /
RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/exercises_s2/train.py"]