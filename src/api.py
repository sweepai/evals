"""
This file creates a FastAPI application for making predictions using the trained model.
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

# Load the trained model for digit recognition
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform is a sequence of preprocessing steps applied to the images for prediction
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction for an uploaded image.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Apply the transform to the image
    image = transform(image)
    # Add a batch dimension to the image
    image = image.unsqueeze(0)  
    with torch.no_grad():
        # Pass the image through the model and get the output
        output = model(image)
        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
    # Return the prediction
    return {"prediction": int(predicted[0])}
