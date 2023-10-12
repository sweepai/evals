from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from cnn import CNN  # Import the CNN class

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Instantiate the CNN
model = CNN()  # Instantiate the CNN class

# Step 3: Train the CNN
learning_rate = 0.01  # Define the learning rate
epochs = 3  # Define the number of epochs
model.train(trainloader, learning_rate, epochs)  # Train the CNN

torch.save(model.state_dict(), "mnist_model.pth")