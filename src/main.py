from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class MNISTTrainer:
    """A class for training a PyTorch model on the MNIST dataset."""

    def load_data(self):
        """Load and preprocess the MNIST dataset.

        Returns:
            DataLoader: A DataLoader for the MNIST dataset.
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader

    def define_model(self):
        """Define the PyTorch model.

        Returns:
            Net: The PyTorch model.
        """
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

        return Net()

    def train(self, trainloader, model):
        """Train the model on the MNIST dataset.

        Args:
            trainloader (DataLoader): The DataLoader for the MNIST dataset.
            model (Net): The PyTorch model.
        """
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        epochs = 3
        for epoch in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

        torch.save(model.state_dict(), "mnist_model.pth")

trainer = MNISTTrainer()
trainloader = trainer.load_data()
model = trainer.define_model()
trainer.train(trainloader, model)