FROM scratch AS sandbox_wheels

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DEMO_DB_PATH=/app/var/demo.db
WORKDIR /app
COPY requirements-sandbox.txt constraints-sandbox.txt ./
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=bind,from=sandbox_wheels,target=/wheels \
    set --; if [ -n "$(ls -A /wheels)" ]; then set -- --no-index --find-links /wheels; fi; \
    pip install "$@" -r requirements-sandbox.txt \
    && groupadd --system sandbox \
    && useradd --system --gid sandbox sandbox \
    && mkdir -p /app/var \
    && chown sandbox:sandbox /app/var
COPY cs_agent/ ./cs_agent/
COPY data/kb/ ./data/kb/
USER sandbox
EXPOSE 8000
CMD ["uvicorn", "--factory", "cs_agent.demo:create_demo_app", "--host", "0.0.0.0", "--port", "8000"]
