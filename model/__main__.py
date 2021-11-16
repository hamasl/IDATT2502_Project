import os

import torch

import model.cnn as cnn

if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    x = torch.load(os.path.join(dirname, "x.pt"))
    y = torch.load(os.path.join(dirname, "y.pt"))
    train_amount = int(0.8*x.shape[0])
    x_train, x_test = torch.split(x, train_amount)
    y_train, y_test = torch.split(y, train_amount)
    num_of_classes = int(torch.max(y).item()) + 1

    # Creating classification bias of form [1, 2, 2, ..., 2, 2] to rather alert for false vulnerabilities,
    # than not alert for actual vulnerabilities.
    classification_bias = 2*torch.ones(num_of_classes)
    classification_bias[0] = 1

    mod = cnn.ConvolutionalNeuralNetworkModel(num_of_classes, x.shape[2], x.shape[3])
    mod.train_model(x_train, y_train, x_test, y_test, 100, 1, verbose=True)
    mod.save_model_state()
