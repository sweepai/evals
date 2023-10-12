"""
This module defines a FastAPI application that uses a PyTorch model to make predictions on uploaded images.

The application provides a single endpoint, /predict, which accepts an image file, preprocesses the image, and uses the PyTorch model to make a prediction. The prediction is then returned as a JSON response.
"""

# FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# torchvision is a library that has datasets, model architectures and image transformation tools
from torchvision import transforms
# Importing Net class from main.py which defines the PyTorch model used for making predictions
from main import Net  

# Load the PyTorch model defined in the Net class from main.py
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Define the transformation pipeline used for preprocessing the image. The transforms.Compose function is used to chain together the ToTensor and Normalize transformations.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# FastAPI application that provides the /predict endpoint
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function accepts an image file, preprocesses the image, and uses the PyTorch model to make a prediction.

    Parameters:
    file (UploadFile): The image file to be processed.

    Returns:
    dict: A dictionary with a single key, 'prediction', whose value is the prediction made by the model.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
