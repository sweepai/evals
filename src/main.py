from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class MNISTTrainer:
    def __init__(self, epochs=3):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.trainset = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        self.trainloader = DataLoader(self.trainset, batch_size=64, shuffle=True)
        self.testset = datasets.MNIST('.', download=True, train=False, transform=self.transform)
        self.testloader = DataLoader(self.testset, batch_size=64, shuffle=True)
        self.model = self._define_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.NLLLoss()
        self.epochs = epochs

    def _define_model(self):
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

    def train(self):
        for epoch in range(self.epochs):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

        torch.save(self.model.state_dict(), "mnist_model.pth")

trainer = MNISTTrainer()
trainer.train()

torch.save(model.state_dict(), "mnist_model.pth")