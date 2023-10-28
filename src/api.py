"""
This script defines a FastAPI application that uses the PyTorch model defined in main.py to make predictions on uploaded images.
"""

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from main import Net  # Importing Net class from main.py

# Load the model
# The model is loaded from the saved state dictionary and set to evaluation mode with model.eval()
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image
# The transformation pipeline consists of two steps:
# 1. transforms.ToTensor() - Converts the input image to PyTorch tensor.
# 2. transforms.Normalize((0.5,), (0.5,)) - Normalizes the tensor with mean 0.5 and standard deviation 0.5.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function takes an uploaded file as input, preprocesses the image, and makes a prediction using the PyTorch model.
    The prediction is returned as a dictionary with the key 'prediction'.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Apply the transformation pipeline to the image
    image = transform(image)
    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)  # Add batch dimension
    # Make a prediction with the model
    with torch.no_grad():
        output = model(image)
        # Get the class with the highest probability
        _, predicted = torch.max(output.data, 1)
    # Return the prediction as a dictionary
    return {"prediction": int(predicted[0])}
