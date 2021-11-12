import os

import torch
import torch.nn as nn


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self, num_of_classes: int, input_element_size: int, encoding_size_per_element: int,
                 device=torch.device("cpu"), directory="state"):
        """
        Creates a model object using the given parameters.

        :param num_of_classes: The number of classes each element can be classified as.
        :param input_element_size: The size of each input element. All input elements need to be of the same size.
        :param device: Is the device that the neural network should run on. By default it's cpu but it can be changed to cuda.
        """
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        self.num_of_classes = num_of_classes
        self.input_element_size = input_element_size
        self.encoding_size_per_element = encoding_size_per_element
        self.device = device
        self._dirname = os.path.join(os.path.dirname(__file__), directory)

        # Model layers (includes initialized model variables):
        self.logits = self._get_model()

    def _get_model(self):
        """
        Create a sequential model.
        :return: None
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * (self.input_element_size // 4) * (self.encoding_size_per_element // 4), 1024),
            nn.Linear(1024, self.num_of_classes)).to(self.device)

    # Predictor
    def f(self, x: torch.Tensor):
        """
        Returns a prediction of what classes the elements belong to.
        :param x: The tensor to make a prediction from.
        :return: torch.Tensor
        """
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculates cross entropy loss of x and y tensor.
        :param x: The x data.
        :param y: The y data.
        :return: Tensor containing the loss.
        """
        # TODO add documentation
        return nn.functional.cross_entropy(self.logits(x), y).to(self.device)

    # Accuracy
    def accuracy(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        """
        Measures accuracy by taking in data elements, and the actual classes to those data points
        :param x: The data elements to be predicted.
        :param y: The correct class for each data element
        :param batch_size: The size of each batch
        :return: torch.Tensor
        """

        accuracy = 0
        x_batches = torch.split(x, batch_size)
        y_batches = torch.split(y, batch_size)
        for i in range(len(x_batches)):
            accuracy += torch.mean(torch.eq(self.f(x_batches[i].to(self.device)).argmax(1).to(self.device),
                                            y_batches[i].to(self.device)).float()).to(self.device)
        return accuracy / len(x_batches)

    def train_model(self, x_train: torch.Tensor, y_train: torch.Tensor, x_test, y_test, batches=600, epochs=5,
                    learning_rate=0.001,
                    verbose=False):
        """
        Trains the model.
        :param x_train: The x values of the training data.
        :param y_train: The y values of the training data.
        :param batches: The number of bathes that the data should be divided into.
        :param epochs: The number of epochs that the data should be trained for.
        :param learning_rate: Decides how much the model "jump" for each time the data is being optimized. High learning rate may jump over minimas, while lower learning rate may get stuck a local minima.
        :param verbose: If the number of epochs completed should be printed to the console.
        :return: None
        """
        # Divide training data into batches to speed up optimization
        x_train_batches = torch.split(x_train, batches)
        y_train_batches = torch.split(y_train, batches)

        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        for epoch in range(epochs):
            for batch in range(len(x_train_batches)):
                self.loss(x_train_batches[batch].to(self.device),
                          y_train_batches[batch].to(self.device)).backward()  # Compute loss gradients
                optimizer.step()  # Perform optimization by adjusting W and b,
                # Is faster than optimizer.zero_grad: https://betterprogramming.pub/how-to-make-your-pytorch-code-run-faster-93079f3c1f7b
                for param in self.parameters():
                    param.grad = None
            if verbose:
                print(
                    f"Completed {epoch + 1} epochs. Accuracy: {self.accuracy(x_test.to(self.device), y_test.to(self.device), 50)}")

    def save_model_state(self):
        """
        Saves the model state
        :return: None
        """
        with open(os.path.join(self._dirname, "hyper_params.txt"), 'w') as f:
            f.write(f"{self.num_of_classes},{self.input_element_size},{self.encoding_size_per_element}")
        torch.save(self.state_dict(), os.path.join(self._dirname, "cnn_state.pth"))

    def load_model_state(self):
        """
        Loads the model state
        :return: None
        """
        with open(os.path.join(self._dirname, "hyper_params.txt"), 'r') as f:
            line = f.readline().split(",")
            self.num_of_classes = int(line[0])
            self.input_element_size = int(line[1])
            self.encoding_size_per_element= int(line[2])
        self.logits = self._get_model()
        self.load_state_dict(torch.load(os.path.join(self._dirname, "cnn_state.pth")))
