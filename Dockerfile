FROM python:3.10-slim as base

ARG DOCKER_USER=devuser

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV TZ="Europe/Berlin"

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN useradd -ms /bin/bash $DOCKER_USER
RUN usermod -aG sudo $DOCKER_USER
USER $DOCKER_USER

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


FROM base as uv-deps

ARG DOCKER_USER=devuser
ARG APP_DIR=app
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

WORKDIR /$DOCKER_USER/$APP_DIR
COPY . .
RUN uv sync --locked --no-install-project --dev

CMD ["PYTHONPATH='${PYTHONPATH}:${PWD}' uv run tests/test_inference.py"]