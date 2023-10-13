"""
This module is used to create a FastAPI application that serves a machine learning model trained on the MNIST dataset.
"""

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library
import torch
# torchvision is a library for PyTorch that provides access to popular datasets, model architectures, and image transformations for computer vision
from torchvision import transforms
# Importing Net class from main.py
from main import Net  

# Load the model. This is the trained neural network model that we will use for making predictions.
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image. The transforms are converting the data to PyTorch tensors and normalizing the pixel values.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# FastAPI application. This is the main entry point for our API.
app = FastAPI()

# This function is used to make predictions on uploaded images.
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction on an uploaded image.

    Parameters
    ----------
    file : UploadFile
        The image file to make a prediction on.

    Returns
    -------
    dict
        A dictionary with a single key 'prediction' and the predicted digit as the value.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
