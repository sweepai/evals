import unittest
from unittest.mock import Mock, patch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import Net

class TestMain(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=Net)
        self.mock_model.forward.return_value = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.mock_data_loader = Mock(spec=DataLoader)
        self.mock_data_loader.return_value = [torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,))]

    def test_model_initialization(self):
        model = Net()
        self.assertIsInstance(model, Net)

    def test_model_forward_pass(self):
        input_data = torch.randn(64, 1, 28, 28)
        output = self.mock_model.forward(input_data)
        self.assertEqual(output.shape, (64, 10))

    def test_data_loader(self):
        batch = next(iter(self.mock_data_loader))
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].shape, (64, 1, 28, 28))
        self.assertEqual(batch[1].shape, (64,))

if __name__ == "__main__":
    unittest.main()
