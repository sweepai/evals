import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST dataset.
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train(self, dataloader, optimizer):
        criterion = nn.CrossEntropyLoss()
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = self.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def predict(self, image):
        output = self.forward(image)
        _, predicted = torch.max(output.data, 1)
        return predicted.item()
