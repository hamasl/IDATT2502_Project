import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import random_split, TensorDataset
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self, num_of_classes: int, input_element_size: int, encoding_size_per_element: int,
                 device=torch.device("cpu"), directory="state", class_names: str = None, classification_bias: Tensor = None):
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
        # Does not matter if classifiaction_bias is none because cross entropy loss with None as weights behave as without weights.
        self.classification_bias = classification_bias
        self.class_names = class_names

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
        return nn.functional.cross_entropy(self.logits(x), y, weight=self.classification_bias.to(self.device)).to(self.device)

    # Accuracy
    def accuracy(self, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        """
        Measures accuracy by taking in data elements, and the actual classes to those data points
        :param x: The data elements to be predicted.
        :param y: The correct class for each data element.
        :param batch_size: The size of each batch.
        :return: float
        """

        accuracy = 0
        x_batches = torch.split(x, batch_size)
        y_batches = torch.split(y, batch_size)
        for i in range(len(x_batches)):
            accuracy += torch.mean(torch.eq(self.f(x_batches[i].to(self.device)).argmax(1).to(self.device),
                                            y_batches[i].to(self.device)).float()).to(self.device)

        return accuracy / len(x_batches)

    def confusion_matrix(self, x: torch.Tensor, y: torch.Tensor, batch_size: int):
        """
        Creates a confusion matrix from the predictions and the labels
        :param x: The data elements to be predicted.
        :param y: The correct class for each data element.
        :param batch_size: The size of each batch.
        :return: confusion matrix
        """
        predicted_answer = torch.tensor([])
        x_batches = torch.split(x, batch_size)
        y_batches = torch.split(y, batch_size)
        for i in range(len(x_batches)):
            batch_predict = self.f(x_batches[i].to(self.device)).argmax(1).to(torch.device("cpu"))
            predicted_answer = torch.cat((predicted_answer, batch_predict), 0)
        ConfusionMatrixDisplay.from_predictions(y.to(torch.device("cpu")), predicted_answer, display_labels=self.class_names, 
                                                normalize='true', xticks_rotation='vertical')
        os.make
        plt.savefig(os.path.join(self._dirname, "plots"))

    def false_positive_vs_false_negative(self, x: torch.Tensor, y: torch.Tensor, batch_size: int):
        false_positives = [0] * self.num_of_classes
        false_negatives = [0] * self.num_of_classes

        x_batches = torch.split(x, batch_size)
        y_batches = torch.split(y, batch_size)
        for i in range(len(x_batches)):
            predictions = self.f(x_batches[i].to(self.device)).argmax(1).to(self.device)
            for j in range(len(predictions)):
                pred = predictions[j].item()
                ans = y_batches[i][j].item()
                if pred != ans:
                    false_positives[pred] += 1
                    false_negatives[ans] += 1
        return false_positives, false_negatives

    def split_data(self, x: torch.Tensor, y: torch.Tensor):
        """
        Splits the data into train and test set
        :param x: The x tensor, with padded word2vec vectors
        :param y: The y tensor, with classes
        """
        data_set = TensorDataset(x, y)
        train_set_size = int(len(x) * 0.8)
        valid_set_size = len(x) - train_set_size
        train_set, test_set = random_split(data_set, [train_set_size, valid_set_size])
        x_train, y_train = train_set[:][0], train_set[:][1]
        x_test, y_test = test_set[:][0], test_set[:][1]
        return x_train, y_train, x_test, y_test

    def train_model(self, x: torch.Tensor, y: torch.Tensor, batch_size=600, cross_validations=1,
                    learning_rate=0.001, epochs=5, verbose=False):
        """
        Trains the model.
        :param x: The raw input of the x tensor
        :param y: The raw input of the y tensor
        :param batch_size: The size of each batch.
        :param epochs: Number of epochs per cross validation
        :param cross_validations: The number of cross validations that the data should be trained for.
        :param learning_rate: Decides how much the model "jump" for each time the data is being optimized. High learning rate may jump over minimas, while lower learning rate may get stuck a local minima.
        :param verbose: If the number of epochs completed should be printed to the console.
        :return: None
        """
        for cross_validation in range(cross_validations):
            x_train, y_train, x_test, y_test = self.split_data(x, y)

            # Divide training data into batches to speed up optimization
            x_train_batches = torch.split(x_train, batch_size)
            y_train_batches = torch.split(y_train, batch_size)

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
                        f"Completed {(epoch + 1) + (cross_validation * epochs)} epochs. Accuracy: {self.accuracy(x_test.to(self.device), y_test.to(self.device), 50)}")
                    if(epoch+1 == epochs): self.confusion_matrix(x_test.to(self.device), y_test.to(self.device), 20)
        x_train, y_train, x_test, y_test = self.split_data(x, y)
        false_positives, false_negatives = self.false_positive_vs_false_negative(x_test, y_test, 100)
        fig, axs = plt.subplots(1, len(false_negatives), sharey=True)
        # fig.title("FP vs FN grouped by class.")
        for i in range(len(false_negatives)):
            total = false_positives[i] + false_negatives[i]
            axs[i].title.set_text(i if self.class_names is None else self.class_names[i])
            axs[i].bar("FP", 100 * false_positives[i] / total, color="blue", label="FP = False positive")
            axs[i].bar("FN", 100 * false_negatives[i] / total, color="orange", label="FN = False negative")
        handles, labels = axs[len(false_negatives) - 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
        plt.show()

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
            self.encoding_size_per_element = int(line[2])
        self.logits = self._get_model()
        self.load_state_dict(torch.load(os.path.join(self._dirname, "cnn_state.pth"), map_location=self.device))
