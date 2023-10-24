import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.cnn import (  # Import CNN class and train_cnn function from cnn.py
    CNN,
    train_cnn,
)

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
# Create an instance of the CNN class
model = CNN()  # Create an instance of the CNN class

# Step 3: Train the Model
# Call the train_cnn function, passing the CNN instance, the DataLoader instance, and a number of epochs
train_cnn(
    model, trainloader, 10
)  # Train the CNN model with the trainloader data for 10 epochs

torch.save(model.state_dict(), "mnist_model.pth")  # Save the trained model
