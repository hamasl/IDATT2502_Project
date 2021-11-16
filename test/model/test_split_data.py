import unittest
import torch

import model.cnn as cnn


class SplitTest(unittest.TestCase):

    def test_correct_splitting(self):
        input_size = 24
        num_of_elements = 600
        element_encoding = 20
        model = cnn.ConvolutionalNeuralNetworkModel(10, input_size, element_encoding)
        input_x = torch.tensor([1, 2, 3, 4, 5, 6])
        input_y = torch.tensor([1, 2, 3, 4, 5, 6])
        x_train, y_train, x_test, y_test = model.split_data(input_x, input_y)
        self.assertListEqual(x_train.tolist(), y_train.tolist())
        self.assertListEqual(x_test.tolist(), y_test.tolist())

