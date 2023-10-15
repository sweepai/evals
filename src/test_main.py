from unittest import mock

import pytest

from main import Net, trainloader, trainset


def test_data_loading_and_preprocessing():
    """
    Test the data loading and preprocessing steps.
    """
    with mock.patch("torch.utils.data.DataLoader") as mock_dataloader:
        mock_dataloader.return_value = trainloader
        pytest.assume(mock_dataloader.called)
        pytest.assume(mock_dataloader.call_args[0][0] == trainset)
        pytest.assume(mock_dataloader.call_args[1]["batch_size"] == 64)
        pytest.assume(mock_dataloader.call_args[1]["shuffle"] is True)


def test_model_definition():
    """
    Test the model definition.
    """
    with mock.patch("torch.nn.Module") as mock_module:
        mock_module.return_value = Net()
        pytest.assume(mock_module.called)
        pytest.assume(isinstance(mock_module.return_value, Net))
        pytest.assume(hasattr(mock_module.return_value, "fc1"))
        pytest.assume(hasattr(mock_module.return_value, "fc2"))
        pytest.assume(hasattr(mock_module.return_value, "fc3"))
