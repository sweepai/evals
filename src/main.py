"""
This module is used to train a simple neural network model on the MNIST dataset using PyTorch.
"""

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Step 1: Load MNIST Data and Preprocess
# Compose transforms to convert the images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the tensor
])

# Download the MNIST dataset and apply the transforms
trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
# Create a DataLoader to handle batching of the MNIST dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    A simple feed-forward neural network with one hidden layer.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer, from 784 nodes to 128
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer, from 128 nodes to 64
        self.fc3 = nn.Linear(64, 10)  # Output layer, from 64 nodes to 10 (for the 10 classes)
    
    def forward(self, x):
        """
        Defines the forward pass of the network.
        """
        x = x.view(-1, 28 * 28)  # Flatten the input tensor
        x = nn.functional.relu(self.fc1(x))  # Pass through the first layer and apply ReLU activation function
        x = nn.functional.relu(self.fc2(x))  # Pass through the second layer and apply ReLU activation function
        x = self.fc3(x)  # Pass through the output layer
        return nn.functional.log_softmax(x, dim=1)  # Apply log softmax to the output

# Step 3: Train the Model
# Initialize the model, optimizer and loss function
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()  # Zero the gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

# Save the trained model
torch.save(model.state_dict(), "mnist_model.pth")