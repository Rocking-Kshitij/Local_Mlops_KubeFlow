FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements1.txt .
RUN pip install --no-cache-dir -r requirements1.txt

ARG MLFLOW_TRACKING_URI
ARG MLFLOW_S3_ENDPOINT_URL
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
ENV MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}

# Copy code and install dependencies
COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements2.txt

COPY train_mnist_mlflow.py .

# Set default command
CMD python train_mnist_mlflow.py
