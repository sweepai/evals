"""
This module contains code for training a simple feed-forward neural network on the MNIST dataset using PyTorch. 
The MNIST dataset consists of 28x28 grayscale images of handwritten digits. 
The PyTorch model defined in this module is a simple feed-forward neural network with two hidden layers. 
The data is preprocessed by converting the images to tensors and normalizing them.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library based on the Torch library
import torch
# torch.nn is a sublibrary of PyTorch, provides classes to build neural networks
import torch.nn as nn
# torch.optim is a package implementing various optimization algorithms
import torch.optim as optim
# torchvision is a library for PyTorch that provides datasets and models for computer vision
from torchvision import datasets, transforms
# DataLoader combines a dataset and a sampler, and provides an iterable over the given dataset
from torch.utils.data import DataLoader
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# transforms.Compose composes several transforms together, ToTensor converts a PIL Image or numpy.ndarray to tensor, 
# and Normalize normalizes a tensor image with mean and standard deviation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# datasets.MNIST loads the MNIST dataset from root directory with train/test argument, applies the transformations and downloads the data if not available
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# DataLoader wraps an iterable around the Dataset to enable easy access to the samples
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class defines a simple feed-forward neural network with two hidden layers.
    The network takes as input a 28x28 image, flattens it into a 784-dimensional vector, 
    and then passes it through two hidden layers with ReLU activation, 
    followed by an output layer with log softmax activation.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.
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