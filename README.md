# evals

This project now includes a new class `MNISTTrainer` which is used to train a model on the MNIST dataset.

## MNISTTrainer

The `MNISTTrainer` class is defined in `src/main.py`. It includes methods for loading and preprocessing the MNIST dataset, defining the model architecture, and training the model.

### Usage

An instance of `MNISTTrainer` is created and then its methods are used to load the data, define the model, and train the model. Here is an example:

```python
# Create an instance of MNISTTrainer
trainer = MNISTTrainer()

# Load the data
trainloader = trainer.load_data()

# Define the model
Net = trainer.define_model()
model = Net(trainloader)

# Train the model
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    for images, labels in model.trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

The trained model is then saved and can be loaded in `src/api.py` using the `MNISTTrainer` class in a similar way:

```python
# Create an instance of MNISTTrainer
trainer = MNISTTrainer()

# Load the data
trainloader = trainer.load_data()

# Define the model
Net = trainer.define_model()
model = Net(trainloader)

# Load the model's state from the saved file
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()
```