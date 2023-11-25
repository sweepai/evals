import torch
from fastapi import FastAPI, File, UploadFile
from main import Trainer  # Importing Trainer class from main.py
from PIL import Image
from torchvision import transforms

# Load the model
# Initialize the Trainer with the appropriate model and parameters (assuming defaults)
trainer = Trainer(model=Net, args=[], optimizer=optim.SGD, optim_args={'lr': 0.01}, loss_fn=nn.NLLLoss())
trainer.load_model("mnist_model.pth")
model = trainer.model
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
