import unittest
from unittest.mock import Mock
from main import Net
import torch

class TestNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.net = Net()

    def test_net_init(self):
        self.assertIsInstance(self.net.fc1, torch.nn.Linear)
        self.assertIsInstance(self.net.fc2, torch.nn.Linear)
        self.assertIsInstance(self.net.fc3, torch.nn.Linear)

    def test_net_forward(self):
        mock_tensor = torch.randn(784)
        output = self.net(mock_tensor)
        self.assertIsInstance(output, torch.Tensor)

    def test_training_loop(self):
        mock_images = torch.randn(64, 784)
        mock_labels = torch.randint(0, 10, (64,))
        mock_trainloader = [(mock_images, mock_labels) for _ in range(10)]
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()
        for images, labels in mock_trainloader:
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        self.assertIsNotNone(model.state_dict())

if __name__ == '__main__':
    unittest.main()
