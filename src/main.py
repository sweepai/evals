"""
This module is used to train a simple neural network on the MNIST dataset using PyTorch.
"""

# PIL is used for opening, manipulating, and saving many different image file formats
from PIL import Image
# PyTorch is an open source machine learning library
import torch
# nn is a module of PyTorch that provides classes for building neural networks
import torch.nn as nn
# optim is a module that implements various optimization algorithms
import torch.optim as optim
# torchvision is a library for image and video processing
from torchvision import datasets, transforms
# DataLoader is a utility class that provides the ability to batch, shuffle and load data in parallel
from torch.utils.data import DataLoader
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# The transforms are used to preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# trainset represents the MNIST dataset
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# trainloader is a DataLoader instance that provides the ability to batch, shuffle and load the trainset in parallel
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class represents a simple neural network with one hidden layer.
    
    The forward method implements the forward pass of the neural network.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 28 * 28)
        # Apply the first fully connected layer and the ReLU activation function
        x = nn.functional.relu(self.fc1(x))
        # Apply the second fully connected layer and the ReLU activation function
        x = nn.functional.relu(self.fc2(x))
        # Apply the third fully connected layer
        x = self.fc3(x)
        # Apply the log softmax activation function
        return nn.functional.log_softmax(x, dim=1)

# Step 3: Train the Model
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    # Iterate over the batches of images and labels
    for images, labels in trainloader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass: compute the output of the model on the input images
        output = model(images)
        # Compute the loss of the output
        loss = criterion(output, labels)
        # Backward pass: compute the gradients of the loss with respect to the model's parameters
        loss.backward()
        # Update the model's parameters
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")