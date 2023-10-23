import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

from main import MNISTTrainer  # Importing MNISTTrainer class from main.py

# Load the model
trainer = MNISTTrainer()
model = trainer.define_model()
trainer.load()
model.eval()

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
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
