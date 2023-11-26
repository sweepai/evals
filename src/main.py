import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN, train
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)








    








model = Net()
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
train(model, trainloader, optimizer)


torch.save(model.state_dict(), "mnist_model.pth")