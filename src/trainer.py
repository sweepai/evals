import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, trainloader, lr, epochs):
        self.model = model
        self.trainloader = trainloader
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = nn.NLLLoss()

    def train(self):
        for epoch in range(self.epochs):
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
        torch.save(self.model.state_dict(), "mnist_model.pth")
