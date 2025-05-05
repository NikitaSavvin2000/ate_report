FROM python:3.10.1-buster

RUN apt-get update && apt-get install -y vim poppler-utils

WORKDIR /app

# Копируем всё содержимое проекта (кроме указанного в .dockerignore)
COPY . .

RUN pip install -U pip setuptools wheel && \
    pip install pdm && \
    pdm install --prod --no-lock --no-editable && \
    pdm build && \
    pdm install

EXPOSE 8501

ENTRYPOINT ["pdm", "run", "src/server.py"]