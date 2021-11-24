import os

import torch

import model.cnn as cnn
import preprocessing.class_names as cn


if __name__ == '__main__':
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    x = torch.load(os.path.join(dirname, "x.pt"))
    y = torch.load(os.path.join(dirname, "y.pt"))
    num_of_classes = int(torch.max(y).item()) + 1

    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2], x.shape[3], class_names=cn.class_names)
    mod.train_model(x, y, batch_size=50, cross_validations=1, epochs=1, verbose=True)
    mod.save_model_state()
