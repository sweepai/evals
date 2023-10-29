"""
This module is used to train a simple neural network model on the MNIST dataset using PyTorch.
It includes steps for loading and preprocessing the dataset, defining the model, and training the model.
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# Define the transformations to be applied on the images
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize the images
])

# Load the MNIST dataset and apply the transformations
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)

# Create a DataLoader for the dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class defines a simple feed-forward neural network model.
    
    Methods
    -------
    forward(x)
        Defines the forward pass of the model.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # Third fully connected layer
    
    def forward(self, x):
        """
        Defines the forward pass of the model.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            
        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = nn.functional.relu(self.fc1(x))  # Apply ReLU activation function after the first layer
        x = nn.functional.relu(self.fc2(x))  # Apply ReLU activation function after the second layer
        x = self.fc3(x)  # Apply the third layer
        return nn.functional.log_softmax(x, dim=1)  # Apply log softmax to the output

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