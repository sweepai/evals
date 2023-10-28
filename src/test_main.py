import pytest
from unittest.mock import Mock
from main import Net
import torch

@pytest.fixture
def net():
    return Net()

def test_net_init(net):
    assert isinstance(net.fc1, torch.nn.Linear)
    assert isinstance(net.fc2, torch.nn.Linear)
    assert isinstance(net.fc3, torch.nn.Linear)

def test_net_forward(mocker, net):
    mock_tensor = mocker.Mock(spec=torch.Tensor)
    mock_tensor.view.return_value = mock_tensor
    output = net(mock_tensor)
    assert isinstance(output, torch.Tensor)

def test_training_loop(mocker):
    mock_images = mocker.Mock(spec=torch.Tensor)
    mock_labels = mocker.Mock(spec=torch.Tensor)
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
    assert model.state_dict() is not None
