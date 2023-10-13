import pytest
from unittest.mock import Mock, patch
from src import main

def test_transform():
    mock_image = Mock(spec=Image.Image)
    tensor = main.transform(mock_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 28, 28)
    assert tensor.min() >= -1
    assert tensor.max() <= 1

@patch('torchvision.datasets.MNIST')
@patch('torch.utils.data.DataLoader')
def test_trainloader(mock_dataloader, mock_mnist):
    mock_mnist.return_value = Mock()
    mock_dataloader.return_value = Mock()
    trainloader = main.trainloader
    mock_mnist.assert_called_once_with('.', download=True, train=True, transform=main.transform)
    mock_dataloader.assert_called_once_with(mock_mnist.return_value, batch_size=64, shuffle=True)
    assert trainloader == mock_dataloader.return_value
