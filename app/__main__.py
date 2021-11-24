import sys
import os

import torch

import model.cnn as cnn
import preprocessing.__main__ as preprocess
import preprocessing.class_names as cn
from preprocessing.padding import pad

if __name__ == '__main__':
    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        raise Exception("File not found")
    x, function_names = preprocess.pre_process_predict(file_path)
    # Creating classification bias of form [1, 2, 2, ..., 2, 2] to rather alert for false vulnerabilities,
    # than not alert for actual vulnerabilities.
    # TODO maybe change 2 to 8
    # classification_bias = 8 * torch.ones(len(cn.class_names))
    # classification_bias[0] = 1
    mod = cnn.ConvolutionalNeuralNetworkModel(len(cn.class_names), 0, 0, class_names=cn.class_names)
    mod.load_model_state()
    x = pad(x, mod.encoding_size_per_element, pad_length=mod.input_element_size)
    x_shape = x.shape
    x = torch.reshape(x, (x_shape[0], 1, x_shape[1], x_shape[2]))
    predictions = mod.f(x)
    for i in range(len(predictions)):
        print(f"\n{function_names[i]}:")
        for j in range(len(predictions[i])):
            print(f"{cn.class_names[j]}: {predictions[i, j]}")
