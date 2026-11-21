# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.13.9@sha256:d3b601f0c7b118bc1753febf7c41cc80981df678445e0254284eecf84db79a7d AS uv

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv build --wheel --out-dir /dist

FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS runtime

RUN groupadd --system paircraft \
    && useradd --system --gid paircraft --create-home paircraft

COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore /tmp/*.whl \
    && rm /tmp/*.whl

USER paircraft

ENTRYPOINT ["paircraft"]
CMD ["--version"]
