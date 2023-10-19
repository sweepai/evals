import torch
import torch.nn as nn
import torch.optim as optim

def train(model, trainloader):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()

    epochs = 3
    for epoch in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Training loss at epoch {epoch+1}: {running_loss/len(trainloader)}")
