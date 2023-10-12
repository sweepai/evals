import os
import subprocess
import sys

# Check if pip is installed
try:
    subprocess.run(["pip", "--version"], check=True)
except subprocess.CalledProcessError:
    print("Error: pip is not installed. Please install pip and try again.")
    sys.exit(1)

# Check if poetry is installed, if not, install it
try:
    subprocess.run(["poetry", "--version"], check=True)
except subprocess.CalledProcessError:
    # Download the get-poetry.py script
    subprocess.run(["curl", "-sSL", "https://install.python-poetry.org", "-o", "get-poetry.py"], check=True)
    # Execute the get-poetry.py script to install poetry
    subprocess.run(["python", "get-poetry.py", "--yes"], check=True)

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import MNISTTrainer  # Importing MNISTTrainer class from main.py

# Create an instance of MNISTTrainer and load the model
trainer = MNISTTrainer()
trainer.model.load_state_dict(torch.load("mnist_model.pth"))
trainer.model.eval()

# Transform used for preprocessing the image
transform = trainer.transform

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = trainer.model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
