import to
from PIL import Image


class Trainer:
    def __init__(self, net_class, optimizer_class, criterion_class):
        self.model = net_class()
        self.optimizer = optimizer_class(self.model.parameters(), lr=0.01)
        self.criterion = criterion_class()

    def train(self, trainloader, epochs):
        for epoch in range(epochs):
            for images, labels in trainloader:
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()

    def save_model(self, path="mnist_model.pth"):
        torch.save(self.model.state_dict(), path)


# Initialize and train the model with Trainer
trainer = Trainer(Net, optim.SGD, nn.NLLLoss)
trainer.train(trainloader, epochs=3)
trainer.save_model()




import to
from PIL import Image
