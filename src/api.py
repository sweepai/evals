"""
This module creates a FastAPI application for making predictions using the PyTorch model defined in main.py.
"""
# FastAPI is used for creating the API
from fastapi import FastAPI, UploadFile, File
# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# torch is the main PyTorch library
import torch
# torchvision.transforms provides classes for transforming images
from torchvision import transforms
# Net is the PyTorch model defined in main.py
from main import Net

# 'model' represents the PyTorch model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# 'transform' is a sequence of transformations applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 'app' represents the FastAPI application
app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    This function takes an image file as input, preprocesses it, passes it through the model, and returns the model's prediction.
    The input is an image file and the return value is a dictionary with the key 'prediction' and the model's prediction as the value.
    """
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
