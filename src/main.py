from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class MNISTTrainer:
    def __init__(self):
        # Load and preprocess MNIST data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.trainset = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        self.trainloader = DataLoader(self.trainset, batch_size=64, shuffle=True)

    def train(self, model, optimizer, criterion, epochs):
        # Training loop
        for epoch in range(epochs):
            for images, labels in self.trainloader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs} Loss: {loss.item()}')

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

    def train(self, optimizer, criterion, epochs):
        trainer = MNISTTrainer()
        trainer.train(self, optimizer, criterion, epochs)

# Train the model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()
model.train(optimizer, criterion, 3)

torch.save(model.state_dict(), "mnist_model.pth")