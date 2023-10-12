import torch
import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    """
    Convolutional Neural Network (CNN) class for handling the MNIST dataset.
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Implements the forward pass of the CNN.
        """
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

    def train(self, trainloader, epochs=3):
        """
        Trains the CNN on the MNIST dataset.
        """
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        for epoch in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                output = self(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
