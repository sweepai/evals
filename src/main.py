from PIL import Ima


class Trainer:
    def __init__(self, model_class, model_params, optimizer_class, optimizer_params, criterion):
        self.model = model_class(*model_params)
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        self.criterion = criterion

    def train(self, trainloader, epochs):
        for epoch in range(epochs):
            for images, labels in trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{epochs} completed")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

ge
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)

# Step 3: Train the Model
# Initialize Trainer
trainer = Trainer(Net, [], optim.SGD, {'lr': 0.01}, nn.NLLLoss())
# Training loop
epochs = 3
trainer.train(trainloader, epochs)
# Save the trained model
trainer.save_model("mnist_model.pth")