import os

import torch

import model.cnn as cnn

if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    x = torch.load(os.path.join(dirname, "x6.pt"))
    y = torch.load(os.path.join(dirname, "y6.pt"))
    temp = {}
    print(int(torch.max(y).item()) + 1)
    for i in y:
        if i.item() not in temp:
            temp[i.item()] = 1
        else: temp[i.item()] += 1
    print(temp)
    train_amount = int(0.8*x.shape[0])
    x_train, x_test = torch.split(x, train_amount)
    y_train, y_test = torch.split(y, train_amount)
    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2], x.shape[3])
    mod.train_model(x_train, y_train, x_test, y_test, 100, 1, verbose=True)
    mod.save_model_state()
