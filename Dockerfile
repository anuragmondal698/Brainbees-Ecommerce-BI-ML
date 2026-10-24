
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-docker.txt /app/requirements-docker.txt

RUN pip install --no-cache-dir -r /app/requirements-docker.txt \
    && pip install --no-cache-dir --no-deps xgboost==3.4.1

COPY Brainbees_MachineLearning /app/Brainbees_MachineLearning

EXPOSE 10000

CMD ["python", "/app/Brainbees_MachineLearning/src/start_services.py"]

