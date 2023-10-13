from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

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

class MNISTTrainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.criterion = nn.NLLLoss()
        self.epochs = 3

    def load_data(self):
        trainset = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader

    def define_model(self):
        model = Net()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        return model, optimizer

    def train_model(self, trainloader, model, optimizer):
        for epoch in range(self.epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
        return model

    def save_model(self, model):
        torch.save(model.state_dict(), "mnist_model.pth")

trainer = MNISTTrainer()
trainloader = trainer.load_data()
model, optimizer = trainer.define_model()
trained_model = trainer.train_model(trainloader, model, optimizer)
trainer.save_model(trained_model)