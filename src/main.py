"""
This module is used to train a PyTorch model on the MNIST dataset.
It includes the necessary imports, data loading and preprocessing, model definition, and training loop.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load MNIST Data and Preprocess
# Define the transformations to be applied to the images.
# The images are converted to tensors and normalized.
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# Step 2: Define the PyTorch Model
class Net(nn.Module):
    """
    This class defines a simple feed-forward neural network for classifying MNIST images.
    It inherits from the nn.Module class of PyTorch.

    Attributes:
    fc1: First fully connected layer, from input size to 128 nodes.
    fc2: Second fully connected layer, from 128 to 64 nodes.
    fc3: Third fully connected layer, from 64 to 10 nodes (output layer).
    """

    def __init__(self):
        """
        Initializes the network by defining its layers.
        """
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Parameters:
        x: The input tensor.

        Returns:
        The output tensor after passing through the network.
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
# Define the number of training epochs
epochs = 3

# Start the training loop
for epoch in range(epochs):
    # For each batch of images and labels in the trainloader
    for images, labels in trainloader:
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass: compute the output of the model on the images
        output = model(images)
        # Compute the loss between the output and the true labels
        loss = criterion(output, labels)
        # Backward pass: compute the gradients of the loss with respect to the model parameters
        loss.backward()
        # Update the model parameters
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")
