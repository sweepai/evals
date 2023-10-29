"""
This module defines a FastAPI application that serves a model prediction endpoint.
The endpoint accepts an image file, preprocesses the image, and returns a prediction from a pre-trained model.
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Import the Net class from main.py

# Load the pre-trained model
model = Net()  # Initialize the model
model.load_state_dict(torch.load("mnist_model.pth"))  # Load the pre-trained weights
model.eval()  # Set the model to evaluation mode

# Define the transform used for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predict the digit in an uploaded image.

    Parameters
    ----------
    file : UploadFile
        The image file to predict.

    Returns
    -------
    dict
        A dictionary with a single key "prediction" and the predicted digit as the value.
    """
    image = Image.open(file.file).convert("L")  # Open the image file and convert it to grayscale
    image = transform(image)  # Apply the preprocessing transform
    image = image.unsqueeze(0)  # Add a batch dimension to the tensor
    with torch.no_grad():  # Disable gradient computation
        output = model(image)  # Forward pass through the model
        _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability
    return {"prediction": int(predicted[0])}  # Return the prediction as a JSON response
