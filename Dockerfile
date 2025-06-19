# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y git

# Copy code and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train_mnist_mlflow.py .

# Set default command
CMD ["python", "train_mnist_mlflow.py"]

