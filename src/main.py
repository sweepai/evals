"""
This script loads and preprocesses the MNIST dataset, defines a PyTorch model, and trains the model.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# torch.nn is a sublibrary of PyTorch, provides classes to build neural networks
import torch.nn as nn
# torch.optim is a package implementing various optimization algorithms
import torch.optim as optim
# torchvision is a library for PyTorch that provides access to popular datasets, model architectures, and image transformations for computer vision
from torchvision import datasets, transforms
# DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset
from torch.utils.data import DataLoader
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# 'transform' is a sequence of preprocessing steps applied to the images in the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 'trainset' represents the training dataset
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# 'trainloader' is a data loader for batching and shuffling the data
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
# 'Net' class represents a neural network model
class Net(nn.Module):
    """
    A class used to represent a Neural Network

    ...

    Attributes
    ----------
    fc1 : torch.nn.Linear
        First fully connected layer
    fc2 : torch.nn.Linear
        Second fully connected layer
    fc3 : torch.nn.Linear
        Third fully connected layer

    Methods
    -------
    forward(x)
        Defines the computation performed at every call.
    """
    def __init__(self):
        """
        Constructs all the necessary attributes for the neural network object.

        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters:
            x (torch.Tensor): input tensor

        Returns:
            output tensor following the log softmax function
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