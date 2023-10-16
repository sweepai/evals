"""
This script defines and trains a PyTorch model for the MNIST dataset.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load and preprocess the MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert the images to tensors
        transforms.Normalize((0.5,), (0.5,)),  # Normalize the images
    ]
)

# Download the MNIST dataset and apply the transformations
trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
# Create a DataLoader to handle batching of the MNIST dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)


# Define the PyTorch model
class Net(nn.Module):
    """
    This class defines a simple feed-forward neural network for the MNIST dataset.

    The network consists of three fully connected layers. The first two layers use
    the ReLU activation function, and the final layer uses the log softmax function
    for multi-class classification.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = nn.Linear(64, 10)  # Final fully connected layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input images
        x = nn.functional.relu(self.fc1(x))  # Apply ReLU activation function
        x = nn.functional.relu(self.fc2(x))  # Apply ReLU activation function
        x = self.fc3(x)  # Apply log softmax function
        return nn.functional.log_softmax(x, dim=1)


# Initialize the model, optimizer, and loss function
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Train the model
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        optimizer.zero_grad()  # Reset the gradients
        output = model(images)  # Forward pass
        loss = criterion(output, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

# Save the trained model
torch.save(model.state_dict(), "mnist_model.pth")
