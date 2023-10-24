"""
This module defines a FastAPI application that serves a PyTorch model trained on the MNIST dataset.
It includes the necessary imports, model loading, image preprocessing, and prediction endpoint.
"""
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

# Load the trained PyTorch model from the saved state dictionary
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Define the transformations to be applied to the images for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function takes an uploaded image file, preprocesses it, and makes a prediction using the loaded PyTorch model.

    Parameters:
    file: The uploaded image file.

    Returns:
    A dictionary with the key 'prediction' and the predicted digit as the value.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    # Apply the defined transformations to the image
    image = transform(image)
    # Add a batch dimension to the image tensor
    image = image.unsqueeze(0)
    # Make a prediction with the model without computing gradients
    with torch.no_grad():
        output = model(image)
        # Get the digit with the highest prediction score
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
