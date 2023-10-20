from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN

# Step 1: Load MNIST Data and Preprocess
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST(".", download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# Step 2: Define the PyTorch Model
model = CNN()

# Step 3: Train the Model
model.train(trainloader)

# Save the trained model
model.save_model("mnist_model.pth")
