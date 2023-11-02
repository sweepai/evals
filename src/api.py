from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net  # Importing Net class from main.py

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
    """
    This function predicts the digit in an uploaded image.
    It takes as input an image file, preprocesses the image, and then uses a trained model to predict the digit.
    """
    # Open the image file and convert it to grayscale
    image = Image.open(file.file).convert("L")
    
    # Apply the transform to preprocess the image
    image = transform(image)
    
    # Add a batch dimension to the image
    image = image.unsqueeze(0)
    
    # Perform the prediction without calculating gradients
    with torch.no_grad():
        output = model(image)
        # Get the predicted class with the highest output value
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
