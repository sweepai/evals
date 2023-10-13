"""
This module defines a FastAPI application that provides an endpoint for predicting the digit in an uploaded image using a trained neural network model.
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

# Load the trained neural network model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))  # Load the model parameters from a saved state
model.eval()  # Set the model to evaluation mode

# Define the transform used for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the digit in the uploaded image.

    Parameters:
    file (UploadFile): The image file to predict.

    Returns:
    dict: A dictionary with the predicted digit.
    """
    image = Image.open(file.file).convert("L")  # Open the image file and convert it to grayscale
    image = transform(image)  # Apply the preprocessing transform to the image
    image = image.unsqueeze(0)  # Add a batch dimension to the image tensor
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # Forward pass through the model
        _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability
    return {"prediction": int(predicted[0])}  # Return the predicted digit
