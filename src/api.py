"""
This module contains a FastAPI application that serves a prediction endpoint for a simple feed-forward neural network trained on the MNIST dataset.
The prediction endpoint accepts an image file, preprocesses the image, and returns a prediction of the digit in the image.
"""

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# torchvision is a library for PyTorch that provides datasets and models for computer vision
from torchvision import transforms
# Importing Net class from main.py which defines the neural network architecture
from main import Net  

# Load the model. Net class defines the neural network architecture. load_state_dict loads a model’s parameter dictionary using a deserialized state_dict. eval sets the module in evaluation mode.
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image. transforms.Compose composes several transforms together. ToTensor converts a PIL Image or numpy.ndarray to tensor, and Normalize normalizes a tensor image with mean and standard deviation.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function accepts an image file, preprocesses the image, and returns a prediction of the digit in the image.
    The image is opened and converted to grayscale, then transformed to a tensor and normalized.
    The image is then passed through the model to get a prediction.
    
    Parameters:
    file (UploadFile): The image file to predict.

    Returns:
    dict: A dictionary with the key 'prediction' and the predicted digit as the value.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
