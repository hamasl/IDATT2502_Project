import os
import unittest
import torch

import model.cnn as cnn


class CnnTest(unittest.TestCase):

    def test_num_of_classes_is_10_logits_gives_10_classes_per_x_value(self):
        input_size = 24
        num_of_elements = 600
        element_encoding = 20
        model = cnn.ConvolutionalNeuralNetworkModel(10, input_size, element_encoding)
        x = torch.randn(num_of_elements, 1, input_size, element_encoding)
        self.assertEqual(model.logits(x).shape[1], 10)
        self.assertEqual(model.logits(x).shape[0], num_of_elements)

    def test_num_of_classes_is_10_f_gives_10_classes_per_x_value(self):
        input_size = 24
        num_of_elements = 600
        element_encoding = 20
        model = cnn.ConvolutionalNeuralNetworkModel(10, input_size, element_encoding)
        x = torch.randn(num_of_elements, 1, input_size, element_encoding)
        self.assertEqual(model.f(x).shape[1], 10)
        self.assertEqual(model.f(x).shape[0], num_of_elements)

    def test_write_state_and_read_state_to_new_model(self):
        dirname = os.path.join(os.path.dirname(__file__), "state")
        model1 = cnn.ConvolutionalNeuralNetworkModel(10, 32, 30, state_directory=dirname)
        model1.save_model_state()
        model2 = cnn.ConvolutionalNeuralNetworkModel(8, 24, 30, state_directory=dirname)
        model2.load_model_state()
        self.assertEqual(model1.input_element_size, model2.input_element_size)
        self.assertEqual(model1.num_of_classes, model2.num_of_classes)
        self.assertEqual(model1.encoding_size_per_element, model2.encoding_size_per_element)
