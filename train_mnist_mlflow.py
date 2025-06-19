import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv(override=False)  # Will NOT override existing env vars use .env if environment variables are not available



os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")


# 🧠 Model Architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(5408, 10)
        )

    def forward(self, x):
        return self.net(x)

# 🔁 Training_Function
def train_model(epochs=3, lr=0.01, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 🎯 Set MLFlow experiment
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("MNIST-Classifier")

    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("lr", lr)
        mlflow.log_param("batch_size", batch_size)

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

            accuracy = correct / len(train_loader.dataset)
            mlflow.log_metric("loss", total_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.4f}")

            input_tensor = torch.rand(1, 1, 28, 28)
            model.eval()
            with torch.no_grad():
                output_tensor = model(input_tensor)

            # Convert torch tensors to numpy arrays
            input_example = input_tensor.numpy()
            output_example = output_tensor.numpy()

            signature = infer_signature(input_example, output_example)

            # ✅ Log Model with proper input
            mlflow.pytorch.log_model(
                model,
                name="model",
                input_example=input_example,
                signature=signature
            )

if __name__ == "__main__":
    train_model()
