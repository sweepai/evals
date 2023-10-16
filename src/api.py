"""
This script creates an API endpoint using FastAPI. The endpoint accepts an image file,
applies the necessary transformations, and uses a pre-trained PyTorch model to make a prediction.
"""

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from main import Net  # Importing Net class from main.py

# Load the pre-trained model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()  # Set the model to evaluation mode

# Define the transformations to be applied to the input images
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
    ]
)

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function accepts an image file, applies the necessary transformations,
    and uses a pre-trained PyTorch model to make a prediction.

    Parameters:
    file (UploadFile): The image file to be processed.

    Returns:
    dict: A dictionary with the key 'prediction' and the predicted class as the value.
    """
    image = Image.open(file.file).convert("L")  # Convert the image to grayscale
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add a batch dimension
    with torch.no_grad():  # Disable gradient calculation
        output = model(image)  # Make a prediction
        _, predicted = torch.max(
            output.data, 1
        )  # Get the class with the highest probability
    return {"prediction": int(predicted[0])}  # Return the prediction as a dictionary
