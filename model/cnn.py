import torch
import torch.nn as nn


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self, device, num_of_classes, function_size):
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        self._num_of_classes = num_of_classes
        #TODO save function size and num of classes along with weights
        self.function_size = function_size
        self.device = device
        # cnn_multiple is connected to MaxPool layer. Which is kernel_size**num_of_max_pool_layers
        self.cnn_multiple = 4

        # Model layers (includes initialized model variables):
        self.logits = self.get_model()

    def get_model(self):
        if self.function_size % self.cnn_multiple != 0:
            raise Exception(f"Size of each function needs to be a multiple of {self.cnn_multiple}")
        return nn.Sequential(
                                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Flatten(),
                                    nn.Linear(128 * self.function_size//4, 1024),
                                    nn.Linear(1024, self.num_of_classes)).to(self.device)
    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1)).to(self.device)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float()).to(self.device)

    def train_model(self, x_train, y_train, batches = 600, epochs = 20, learning_rate = 0.001, verbose = False):
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
                optimizer.zero_grad()  # Clear gradients for next step

            if verbose:
                print(f"Completed {epoch} epochs.")