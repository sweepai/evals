"""
This module contains the implementation of a neural network model using PyTorch. It includes the data loading and preprocessing step,
definition of the neural network architecture, and the training process for the MNIST dataset. It also contains code to save the trained model.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load MNIST Data and Preprocess
# The 'transform' variable defines a series of preprocessing steps to be applied to the MNIST dataset images.
# It converts images to PyTorch tensors and normalizes them so that the pixel values are in the range [-1, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    The Net class defines a simple feed-forward neural network model with one hidden layer.
    Methods:
        __init__: Initializes the layers of the neural network.
        forward: Computes the forward pass of the network given an input tensor.
    """
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

# Step 3: Train the Model
# 'model' is an instance of the Net class which represents the neural network model.
model = Net()
# 'optimizer' defines the optimization algorithm (Stochastic Gradient Descent) used to update model parameters.
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 'criterion' is the loss function used to measure how well the model performs during training. Here, Negative Log Likelihood Loss is used.
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")