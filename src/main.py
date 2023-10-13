"""
This module is used to train a simple neural network model on the MNIST dataset using PyTorch.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library
import torch
# torch.nn is a sublibrary of PyTorch, used for building and training neural networks
import torch.nn as nn
# torch.optim is a package implementing various optimization algorithms
import torch.optim as optim
# torchvision is a library for PyTorch that provides access to popular datasets, model architectures, and image transformations for computer vision
from torchvision import datasets, transforms
# DataLoader is a PyTorch function for loading and managing datasets
from torch.utils.data import DataLoader
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# The transforms are converting the data to PyTorch tensors and normalizing the pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# The trainset is the MNIST dataset that we will train our model on
# The trainloader is a DataLoader which provides functionality for batching, shuffling and loading data in parallel
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
# The Net class is a definition of a simple neural network
class Net(nn.Module):
    """
    A simple neural network with one hidden layer.

    ...

    Methods
    -------
    forward(x):
        Defines the computation performed at every call.
    """
    def __init__(self):
        """
        Initializes the neural network with one hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : Tensor
            The input data.

        Returns
        -------
        Tensor
            The output of the neural network.
        """
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=1)

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
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