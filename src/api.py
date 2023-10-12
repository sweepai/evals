"""
This module defines an API endpoint for making predictions with the trained model.
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

# Load the model
model = Net()
# Load the trained model parameters and put the model into evaluation mode
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Define the transform used for preprocessing the image
# The transform converts the images to tensors and normalizes them to have a mean of 0.5 and a standard deviation of 0.5.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Make a prediction with the model given an uploaded image.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Apply the transform to the image
    image = transform(image)
    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)
    # Make a prediction with the model
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    # Return the prediction
    return {"prediction": int(predicted[0])}
