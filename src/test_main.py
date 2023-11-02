import unittest
from unittest.mock import patch
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import Net, transform

class TestMain(unittest.TestCase):
    def setUp(self):
        self.model = Net()
        self.transform = transform

    def test_Net(self):
        # Test the forward pass
        input_tensor = torch.randn(1, 1, 28, 28)
        output = self.model(input_tensor)
        self.assertEqual(output.size(), (1, 10))

    @patch('torchvision.datasets.MNIST')
    @patch('torch.utils.data.DataLoader')
    def test_data_loading(self, mock_dataloader, mock_dataset):
        # Mock the MNIST dataset
        mock_dataset.return_value = datasets.MNIST('.', download=True, train=True, transform=self.transform)
        # Mock the DataLoader
        mock_dataloader.return_value = DataLoader(mock_dataset, batch_size=64, shuffle=True)
        # Assert that the DataLoader is called with the correct arguments
        mock_dataloader.assert_called_with(mock_dataset, batch_size=64, shuffle=True)

if __name__ == '__main__':
    unittest.main()
