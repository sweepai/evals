from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logging.basicConfig(filename='training.log', level=logging.ERROR)

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
logging.basicConfig(filename='training.log', level=logging.ERROR)
import logging

logging.basicConfig(filename='training.log', level=logging.ERROR)

trainset = datasets.MNIST('.', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
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
epochs = 3
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
=======
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1} of {epochs}")
    for images, labels in trainloader:
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
=======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
=======
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1} of {epochs}")
    for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
        =======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        for epoch in range(epochs):
        print(f"Starting epoch {epoch+1} of {epochs}")
=======
for epoch in range(epochs):
    print(f"Starting epoch {epoch+1} of {epochs}")
    for images, labels in trainloader:
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
        =======
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
=======
        print(f"Starting epoch {epoch+1} of {epochs}")
        try:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        except Exception as e:
            logging.exception(f"Error occurred during training in epoch {epoch+1}")

torch.save(model.state_dict(), "mnist_model.pth")