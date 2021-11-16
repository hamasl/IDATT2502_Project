import os

import torch

import model.cnn as cnn

if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    x = torch.load(os.path.join(dirname, "x.pt"))
    y = torch.load(os.path.join(dirname, "y.pt"))
    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2], x.shape[3])
    mod.train_model(x, y, 50, 1, verbose=True)
    mod.save_model_state()
