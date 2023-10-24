import unittest
import torch
from main import Net

class TestNet(unittest.TestCase):
    def test_net_initialization(self):
        model = Net()
        self.assertIsInstance(model.fc1, torch.nn.Linear)
        self.assertIsInstance(model.fc2, torch.nn.Linear)
        self.assertIsInstance(model.fc3, torch.nn.Linear)

    def test_net_forward(self):
        model = Net()
        input_tensor = torch.randn(1, 1, 28, 28)
        output = model(input_tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (1, 10))

    def test_training_loop(self):
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.NLLLoss()

        # Small, fixed dataset
        inputs = torch.randn(10, 1, 28, 28)
        targets = torch.randint(0, 10, (10,))

        # Training loop
        initial_loss = float('inf')
        for _ in range(10):
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            self.assertLess(loss.item(), initial_loss)
            initial_loss = loss.item()

if __name__ == '__main__':
    unittest.main()
