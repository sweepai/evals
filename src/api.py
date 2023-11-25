"""
This module provides an API endpoint for predicting the digit in an uploaded image using a pre-trained PyTorch model.

The API endpoint '/predict/' accepts POST requests with an image file, preprocesses the image, and returns the predicted digit.
"""
import torch
from fastapi import FastAPI, File, UploadFile
from main import Net  # Importing Net class from main.py
from PIL import Image
from torchvision import transforms

# Load the model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
    return {"prediction": int(predicted[0])}
    Parameters:
    - file (UploadFile): The image file to predict.

    Returns:
    - dict: A dictionary with the key 'prediction' and the predicted digit as the value.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
