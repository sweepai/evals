from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from main import Net, TrainModel  # Importing Net and TrainModel classes from main.py

# Load the model
model = Net()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Transform used for preprocessing the image
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# Create an instance of TrainModel
train_model = TrainModel(model, criterion, optimizer, trainloader)
train_model.train(3)

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = train_model.model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
