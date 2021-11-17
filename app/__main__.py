import sys
import os

import torch

import model.cnn as cnn
import preprocessing.__main__ as preprocess

if __name__ == '__main__':
    x = preprocess.pre_process_predict(sys.argv[1])
    print(os.path.join(os.path.join(os.path.dirname(__file__), ".."), sys.argv[1]))
    # Creating classification bias of form [1, 2, 2, ..., 2, 2] to rather alert for false vulnerabilities,
    # than not alert for actual vulnerabilities.
    #TODO maybe change 2 to 8
    classification_bias = 2 * torch.ones(num_of_classes)
    classification_bias[0] = 1
    mod = cnn.ConvolutionalNeuralNetworkModel(int(torch.max(y).item()) + 1, x.shape[2], x.shape[3], classification_bias=classification_bias)
    mod.load_model_state()