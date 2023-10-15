"""
This script is used to create a FastAPI application for making predictions using the PyTorch model defined in main.py.
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

# Load the trained PyTorch model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Define the transformations to be applied to the images that are uploaded to the FastAPI application
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create the FastAPI application
app = FastAPI()

# Route handler for the '/predict' endpoint of the FastAPI application
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Load the uploaded image
    image = Image.open(file.file).convert("L")
    # Apply the transformations
    image = transform(image)
    # Add a batch dimension
    image = image.unsqueeze(0)  
    with torch.no_grad():
        # Make the prediction
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    # Return the prediction
    return {"prediction": int(predicted[0])}
