from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

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

# Define the Trainer class
class Trainer:
    def __init__(self, model, optimizer, criterion, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dataloader = dataloader

    def train_epoch(self):
        for images, labels in self.dataloader:
            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()

    def evaluate(self):
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for images, labels in self.dataloader:
                output = self.model(images)
                loss = self.criterion(output, labels)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_correct += (predicted == labels).sum().item()
        average_loss = total_loss / len(self.dataloader)
        accuracy = total_correct / len(self.dataloader.dataset)
        return average_loss, accuracy

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch()
            average_loss, accuracy = self.evaluate()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss}, Accuracy: {accuracy}')

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Create a Trainer instance and train the model
trainer = Trainer(model, optimizer, criterion, trainloader)
trainer.train(3)

torch.save(model.state_dict(), "mnist_model.pth")