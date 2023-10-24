"""
This module creates a FastAPI application for making predictions using the PyTorch model defined in main.py.
"""

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# torchvision is a library for PyTorch that provides access to popular datasets, model architectures, and image transformations for computer vision
from torchvision import transforms
# Importing Net class from main.py
from main import Net  

# 'model' represents the PyTorch model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# 'transform' is a sequence of transformations applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 'app' represents the FastAPI application
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function predicts the digit in the uploaded image file.

    Parameters:
    - file (UploadFile): The image file to predict.

    Returns:
    - dict: A dictionary with the prediction.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
