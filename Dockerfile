# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.13.8@sha256:5cc1805e2426484abca4ad46ea783bab281fdb225fd82d8ce6d66a81254bac85 AS uv

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv build --wheel --out-dir /dist

FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS runtime

RUN groupadd --system paircraft \
    && useradd --system --gid paircraft --create-home paircraft

COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir --disable-pip-version-check --root-user-action=ignore /tmp/*.whl \
    && rm /tmp/*.whl

USER paircraft

ENTRYPOINT ["paircraft"]
CMD ["--version"]
