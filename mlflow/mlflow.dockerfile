FROM python:3.12-slim

WORKDIR /mlflow

EXPOSE 5001

RUN pip install mlflow==3.6.0
