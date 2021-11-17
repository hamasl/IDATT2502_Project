import os

import torch

import model.cnn as cnn

if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    x = torch.load(os.path.join(dirname, "x.pt"))
    y = torch.load(os.path.join(dirname, "y.pt"))
    num_of_classes = int(torch.max(y).item()) + 1

    # Creating classification bias of form [1, 2, 2, ..., 2, 2] to rather alert for false vulnerabilities,
    # than not alert for actual vulnerabilities.
    classification_bias = 2*torch.ones(num_of_classes)
    classification_bias[0] = 1
    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2], x.shape[3], classification_bias=classification_bias)
    mod.train_model(x, y, batches=50, cross_validations=1, epochs=1, verbose=True)
    mod.save_model_state()
