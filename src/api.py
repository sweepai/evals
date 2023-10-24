"""
This module is used to serve a PyTorch model as a FastAPI service. It includes the necessary steps to load the model, 
preprocess the input image, and make a prediction.
"""

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

# Load the model
# The model is loaded from the saved state dictionary and set to evaluation mode.
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image
# The transform is used to convert the input image to a tensor and normalize it.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function takes an uploaded file as input, preprocesses the image, makes a prediction using the model, 
    and returns the prediction.
    
    Parameters:
    file: The uploaded file containing the image.
    
    Returns:
    A dictionary containing the prediction made by the model.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    
    # Preprocess the image using the transform
    image = transform(image)
    
    # Add a batch dimension to the image
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Make a prediction using the model
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    # Return the prediction
    return {"prediction": int(predicted[0])}
