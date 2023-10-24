import unittest
from main import transform, trainset, trainloader, Net, model, optimizer, criterion

class TestMain(unittest.TestCase):
    def test_dataset_loading_and_preprocessing(self):
        self.assertIsNotNone(trainset, "Dataset not loaded")
        self.assertEqual(len(trainset), 60000, "Incorrect dataset size")
        sample_image, sample_label = trainset[0]
        self.assertEqual(sample_image.shape, (1, 28, 28), "Incorrect image shape")
        self.assertEqual(sample_label, 5, "Incorrect label")

    def test_model_definition(self):
        self.assertIsNotNone(model, "Model not defined")
        self.assertEqual(len(list(model.children())), 3, "Incorrect number of layers")
        self.assertIsInstance(list(model.children())[0], nn.Linear, "First layer is not Linear")
        self.assertIsInstance(list(model.children())[1], nn.Linear, "Second layer is not Linear")
        self.assertIsInstance(list(model.children())[2], nn.Linear, "Third layer is not Linear")

    def test_model_training(self):
        initial_loss = float('inf')
        epochs = 3
        for epoch in range(epochs):
            for images, labels in trainloader:
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                self.assertLessEqual(loss.item(), initial_loss, "Loss did not decrease")
                initial_loss = loss.item()
                loss.backward()
                optimizer.step()

if __name__ == '__main__':
    unittest.main()
