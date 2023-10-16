import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTTrainer:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def load_data(self):
        trainset = datasets.MNIST(
            ".", download=True, train=True, transform=self.transform
        )
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

        model = Net()
        return model

    def train_model(self, trainloader, model, epochs, lr):
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        for epoch in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
        return model

    def run(self, epochs=3, lr=0.01, save_path="mnist_model.pth"):
        trainloader = self.load_data()
        model = self.define_model()
        model = self.train_model(trainloader, model, epochs, lr)
        torch.save(model.state_dict(), save_path)
        return model


trainer = MNISTTrainer()
trainer.run()
