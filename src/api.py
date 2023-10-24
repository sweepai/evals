"""
This module is used to serve a FastAPI application that predicts the digit in an uploaded image using a trained PyTorch model.
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

# Load the trained PyTorch model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the transform used for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the digit in the uploaded image using the trained PyTorch model.
    
    Parameters:
    file: The uploaded image file.
    
    Returns:
    A dictionary with the predicted digit.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    
    # Preprocess the image using the defined transform
    image = transform(image)
    
    # Add a batch dimension to the image
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Predict the digit in the image without computing gradients
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    # Return the prediction as a dictionary
    return {"prediction": int(predicted[0])}
