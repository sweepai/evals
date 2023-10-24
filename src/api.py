"""
This script creates a FastAPI application for making predictions using the PyTorch model defined in main.py.
"""
# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# torch is the main PyTorch library
import torch
# torchvision.transforms provides classes for transforming images
from torchvision import transforms
# Importing Net class from main.py
from main import Net

# Load the trained PyTorch model from a file
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Define a sequence of preprocessing steps to be applied to the input images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the FastAPI application
app = FastAPI()

# Define a route handler for making predictions using the model
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function is a route handler for making predictions using the model.
    It takes an image file as input and returns a prediction.

    Parameters:
    file (UploadFile): The image file to predict.

    Returns:
    dict: A dictionary with the prediction.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
