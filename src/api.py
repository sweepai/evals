"""
This file loads the model, preprocesses the image, and makes predictions using the FastAPI framework.
"""

# PyTorch is an open source machine learning library based on the Torch library
import torch

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, File, UploadFile

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image

# torchvision is a library for PyTorch that provides datasets and models for computer vision
from torchvision import transforms

# Importing Net class from main.py
from main import Net

# Load the model and set it to evaluation mode
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image. It converts the image to tensor and normalizes it.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# FastAPI application
app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function takes an image file as input, preprocesses the image, and makes a prediction using the loaded model.
    Parameters:
    file (UploadFile): The image file to be processed and predicted.
    Returns:
    dict: A dictionary with the prediction result.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Apply the transform to the image
    image = transform(image)
    # Add a batch dimension to the image
    image = image.unsqueeze(0)
    # Make a prediction without calculating gradients
    with torch.no_grad():
        output = model(image)
        # Get the index of the max log-probability
        _, predicted = torch.max(output.data, 1)
    # Return the prediction result
    return {"prediction": int(predicted[0])}
