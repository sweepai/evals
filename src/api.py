"""
This script creates a FastAPI application for making predictions using a pre-trained model.
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

# 'model' represents a pre-trained model that is loaded from a file
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# 'transform' is a sequence of preprocessing steps to be applied to the images before making predictions
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 'app' represents the FastAPI application
app = FastAPI()

# '@app.post("/predict/")' defines a route for making predictions with the model
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function makes a prediction using a pre-trained model.

    Parameters:
        file (UploadFile): The image file to predict.

    Returns:
        dict: A dictionary containing the prediction.
    """
    # Open and convert the image file
    image = Image.open(file.file).convert("L")
    # Apply the transform
    image = transform(image)
    # Add a batch dimension
    image = image.unsqueeze(0)  
    with torch.no_grad():
        # Make a prediction with the model
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    # Return the prediction
    return {"prediction": int(predicted[0])}
