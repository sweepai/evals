# evals

This project provides a Python implementation for training and evaluating a simple neural network on the MNIST dataset using PyTorch.

## MNISTTrainer class

The `MNISTTrainer` class is used to train and evaluate the model. It is defined in `src/main.py`.

### Usage

First, create an instance of the `MNISTTrainer` class:

```python
trainer = MNISTTrainer()
```

You can then train the model using the `train` method:

```python
trainer.train()
```

The trained model's parameters are automatically saved to a file named "mnist_model.pth".

To load the model parameters from this file, use the following code:

```python
trainer.model.load_state_dict(torch.load("mnist_model.pth"))
```

To evaluate the model, you can use the `predict` method in `src/api.py`. This method takes an image file as input and returns the model's prediction.

## Dependencies

This project requires the following Python libraries:

- PyTorch
- torchvision
- numpy
- PIL
- FastAPI

Before installing these dependencies, ensure that `pip` is installed and working correctly. You can verify this by running the following command:

```bash
pip --version
```

If `pip` is not installed or not working correctly, you may need to troubleshoot your Python installation or install `pip` separately.

Once `pip` is working correctly, you can install the `poetry` package manager using `pip`:

```bash
pip install poetry
```

After installing `poetry`, verify that it was installed correctly by checking its version:

```bash
poetry --version
```

If the command does not return a version number, `poetry` was not installed correctly.

If you encounter issues with the `poetry` installation, you may need to troubleshoot your `poetry` installation. Here are some steps you can take:

1. Check if the `poetry` executable is in your system's PATH. You can do this by running the following command:

```bash
which poetry
```

If the `poetry` executable is in your PATH, this command will print its location. If it's not, it won't print anything.

2. If the `poetry` executable is not in your PATH, you need to add it. The process for this varies depending on your operating system and shell, but generally involves adding a line to a shell startup file like `~/.bashrc` or `~/.bash_profile` that exports the `poetry` executable's location to the PATH. You can find more detailed instructions in the `poetry` documentation or by searching online for "add to PATH".

3. If you're still having trouble, you can seek help from the `poetry` community or check the `poetry` documentation for more troubleshooting tips. Here's the link to the `poetry` documentation: https://python-poetry.org/docs/

Once `poetry` is installed, you can install the project dependencies using `poetry`:

```bash
poetry install
```

If you are unable to install `poetry`, you can install the project dependencies directly using `pip`:

```bash
pip install torch torchvision numpy pillow fastapi
```

## Running the project

To run the project, first start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

You can then send a POST request to the `/predict` endpoint with an image file to get the model's prediction.