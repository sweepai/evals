from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class MNISTTrainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def load_data(self):
        trainset = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        return trainloader

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
        return Net()

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

    def train_model(self, trainloader, model):
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
        return model

    def run(self):
        trainloader = self.load_data()
        model = self.define_model()
        model = self.train_model(trainloader, model)
        torch.save(model.state_dict(), "mnist_model.pth")
        return model

trainer = MNISTTrainer()
trainer.run()