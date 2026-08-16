FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install poetry
RUN poetry install
# APP_PORT — overridable at build (`docker build --build-arg APP_PORT=8101`)
# and at run (`docker run -e APP_PORT=8101 -p 8101:8101 ...`).
# `serve()` reads it via pydantic-settings (settings.app_port).
ARG APP_PORT=8001
ENV APP_PORT=${APP_PORT}
EXPOSE ${APP_PORT}
CMD ["poetry", "run", "serve"]
