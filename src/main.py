import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MNISTTrainer:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.trainset = None
        self.trainloader = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.NLLLoss()
        self.epochs = 3

    def load_data(self):
        self.trainset = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        self.trainloader = DataLoader(self.trainset, batch_size=64, shuffle=True)

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

    def define_model(self):
        self.model = self.Net()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self):
        for epoch in range(self.epochs):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

    def save_model(self, path="mnist_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path="mnist_model.pth"):
        self.model.load_state_dict(torch.load(path))

trainer = MNISTTrainer()
trainer.load_data()
trainer.define_model()
trainer.train()
trainer.save_model()