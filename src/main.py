from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn import CNN

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
model = CNN()

# Step 3: Train the Model
def train(model, trainloader):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    epochs = 3
    for epoch in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    return model

model = train(model, trainloader)
torch.save(model.state_dict(), "mnist_cnn_model.pth")