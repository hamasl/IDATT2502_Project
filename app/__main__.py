import sys
import os

import torch

import model.cnn as cnn
import preprocessing.__main__ as preprocess
import preprocessing.class_names as cn

if __name__ == '__main__':
    print(os.path.join(os.path.dirname(__file__), ".."))
    print(os.path.join(os.path.join(os.path.dirname(__file__), ".."), sys.argv[1]))
    x, function_names = preprocess.pre_process_predict(os.path.join(os.path.join(os.path.dirname(__file__), ".."), sys.argv[1]))
    # Creating classification bias of form [1, 2, 2, ..., 2, 2] to rather alert for false vulnerabilities,
    # than not alert for actual vulnerabilities.
    # TODO maybe change 2 to 8
    classification_bias = 100 * torch.ones(len(cn.class_names))
    classification_bias[0] = 1
    mod = cnn.ConvolutionalNeuralNetworkModel(len(cn.class_names), x.shape[2], x.shape[3],
                                              classification_bias=classification_bias)
    mod.load_model_state()
    predictions = mod.f(x)
    for i in range(predictions):
        print(f"\n{function_names[i]}:")
        for j in range(predictions[i]):
            print(f"{cn.class_names[j]}: {predictions[i,j]}")