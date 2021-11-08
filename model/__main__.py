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
    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2])
    mod.train_model(x_train, y_train, x_train.shape[0] // 50, 1, verbose=True)
    print(mod.accuracy(x_test, y_test))
    mod.save_model_state()
