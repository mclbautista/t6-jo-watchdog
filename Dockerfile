# JO Telegram Watchdog production image.
# The t6 dependency is a git URL (pyproject.toml), and the slim image has no git. The build
# stage installs git and builds every wheel; the final stage installs those wheels, so the
# production image carries no git. The wheels are installed by file with --no-deps: resolving
# the git URL requirement again would need git. `pip check` then proves the set is complete.
FROM python:3.12-slim AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md ./
COPY jo_watchdog ./jo_watchdog
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dedicated non-root user (uid 1000). /app/data below is owned by it, and the named
# volume mounted there takes that ownership and mode on first use.
RUN groupadd --system --gid 1000 watchdog \
    && useradd --system --uid 1000 --gid watchdog --home-dir /app --shell /usr/sbin/nologin watchdog

WORKDIR /app

COPY --from=build /wheels /wheels
RUN pip install --no-index --no-deps /wheels/*.whl \
    && pip check \
    && rm -rf /wheels \
    && mkdir -p /app/data \
    && chown watchdog:watchdog /app/data \
    && chmod 0700 /app/data

USER watchdog

HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-m", "jo_watchdog.app", "healthcheck", "--config", "/app/config.yaml"]

ENTRYPOINT ["python", "-m", "jo_watchdog.app", "--config", "/app/config.yaml"]
