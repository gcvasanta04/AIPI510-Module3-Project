# Runs the full project: data cleaning, S3 upload, training, and the API.
#
#   docker build -t airbnb-price .
#   docker run --rm -v "$PWD:/app" -v ~/.aws:/root/.aws:ro airbnb-price pipeline
#   docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" airbnb-price serve
#
# Commands: download | clean | upload | train | pipeline | serve | test
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x scripts/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["scripts/entrypoint.sh"]
CMD ["serve"]
