from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class ModelTrainer:
    def __init__(self, epochs=3, lr=0.01, batch_size=64):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.optimizer = None
        self.criterion = nn.NLLLoss()
        self.trainloader = None

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)

    def define_model(self):
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

        self.model = Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train_model(self):
        for epoch in range(self.epochs):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        # TODO: Implement model evaluation
        pass

    def run(self):
        self.load_data()
        self.define_model()
        self.train_model()
        self.evaluate()
        torch.save(self.model.state_dict(), "mnist_model.pth")

trainer = ModelTrainer()
trainer.run()