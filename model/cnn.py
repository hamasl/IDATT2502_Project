import torch
import torch.nn as nn


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self, num_of_classes: int, input_element_size: int, device=torch.device("cpu")):
        """
        Creates a model object using the given parameters.

        :param num_of_classes: The number of classes each element can be classified as.
        :param input_element_size: The size of each input element. All input elements need to be of the same size. Needs to be a multiple of 4 to make the model work, because of two max pool layers with kernel size of 2.
        :param device: Is the device that the neural network should run on. By default it's cpu but it can be changed to cuda.
        """
        super(ConvolutionalNeuralNetworkModel, self).__init__()
        self._num_of_classes = num_of_classes
        # TODO save function size and num of classes along with weights
        self.input_element_size = input_element_size
        self.device = device
        # cnn_multiple is connected to MaxPool layer. Which is kernel_size**num_of_max_pool_layers
        self.cnn_multiple = 4

        # Model layers (includes initialized model variables):
        self.logits = self._get_model()

    def _get_model(self):
        if self.input_element_size % self.cnn_multiple != 0:
            raise Exception(f"Size of each function needs to be a multiple of {self.cnn_multiple}")
        return nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * self.input_element_size//4, 1024),
            nn.Linear(1024, self._num_of_classes)).to(self.device)

    # Predictor
    def f(self, x: torch.Tensor):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x: torch.Tensor, y: torch.Tensor):
        return nn.functional.cross_entropy(self.logits(x), y).to(self.device)

    # Accuracy
    def accuracy(self, x:torch.Tensor, y:torch.Tensor):
        return torch.mean(torch.eq(self.f(x).argmax(1), y).float()).to(self.device)

    def train_model(self, x: torch.Tensor, y: torch.Tensor, batches=600, epochs=5, learning_rate=0.001,
                    verbose=False):
        """
        Trains the model.
        :param x: The x values of the training data.
        :param y: The y values of the training data.
        :param batches: The number of bathes that the data should be divided into.
        :param epochs: The number of epochs that the data should be trained for.
        :param learning_rate: Decides how much the model "jump" for each time the data is being optimized. High learning rate may jump over minimas, while lower learning rate may get stuck a local minima.
        :param verbose: If the number of epochs completed should be printed to the console.
        :return: None
        """
        # Divide training data into batches to speed up optimization
        x_train_batches = torch.split(x, batches)
        y_train_batches = torch.split(y, batches)

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

    def save_model_state(self, directory: str):
        """
        Saves the model state
        :param directory:
        :return:
        """
        with open(directory + "/hyper_params.txt", 'w') as f:
            f.write(f"{self._num_of_classes},{self.input_element_size}")
        torch.save(self.state_dict(), directory + "/cnn_state.pth")

    def load_model_state(self, directory: str):
        """
        Loads the model state
        :param directory:
        :return:
        """
        with open(directory + "/hyper_params.txt", 'r') as f:
            line = f.readline().split(",")
            self._num_of_classes = int(line[0])
            self.input_element_size = int(line[1])
        self.logits = self._get_model()
        self.load_state_dict(torch.load(directory + "/cnn_state.pth"))


if __name__ == '__main__':
    model = ConvolutionalNeuralNetworkModel(12, 32)
    model.load_model_state("./state")
    print(model.input_element_size)
    print(model._num_of_classes)
    x_train = torch.rand(600, 1, 32)
    print(x_train)
    y_train = torch.randint(0, 9, (600,))
    print(y_train)
    model.train_model(x_train, y_train)
    #model.save_model_state("./state")
