import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from main import MNISTTrainer  # Importing MNISTTrainer class from main.py

# Create an instance of MNISTTrainer and run the training
trainer = MNISTTrainer()
model = trainer.run()

app = FastAPI()


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")
    image = trainer.preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return {"prediction": int(predicted[0])}
