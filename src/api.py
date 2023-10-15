"""
This module is used to create a FastAPI application that serves a model trained on the MNIST dataset.
The application provides an endpoint that accepts an image file, preprocesses the image, and returns a prediction of the digit in the image.
"""

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library
import torch
# torchvision is a library for image and video processing
from torchvision import transforms
# Importing Net class from main.py
from main import Net  

# Load the model trained on the MNIST dataset
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image. It converts the image to a tensor and normalizes it.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function accepts an image file, preprocesses the image, and returns a prediction of the digit in the image.
    
    Parameters:
    file (UploadFile): The image file to be processed.
    
    Returns:
    dict: A dictionary with the key 'prediction' and the predicted digit as the value.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Preprocess the image
    image = transform(image)
    # Add batch dimension
    image = image.unsqueeze(0)  
    with torch.no_grad():
        # Compute the output of the model on the input image
        output = model(image)
        # Get the predicted class
        _, predicted = torch.max(output.data, 1)
    # Return the prediction
    return {"prediction": int(predicted[0])}
