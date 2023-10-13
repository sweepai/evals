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

# Step 2: Define the PyTorch Model
model = CNN()  # Instantiate the CNN class
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Step 3: Train the Model
# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in trainloader:
        images = images.view(-1, 1, 28, 28)  # Reshape the input data for the CNN
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "mnist_model.pth")