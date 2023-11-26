import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from main import Net  # Importing Net class from main.py
from main import Trainer  # Importing Trainer class from main.py

# Load the model
trainer = Trainer()
trainer.load_model("mnist_model.pth")  # Assuming load_model method exists

# Transform used for preprocessing the image
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = trainer.get_model()(image)  # Assuming get_model method exists
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
