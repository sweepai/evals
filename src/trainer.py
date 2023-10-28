import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, dataloader, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = nn.NLLLoss()

    def train(self, epochs):
        for _epoch in range(epochs):
            for images, labels in self.dataloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

        torch.save(self.model.state_dict(), "mnist_model.pth")
