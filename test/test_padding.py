import unittest
import preprocessing.padding as padding


class PaddingTest(unittest.TestCase):

    def test_add_padding(self):
        input = [[[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]],
                 [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]],
                 [[1, 0, 0, 0]]]

        padded = padding.pad(input, 4)
        expected_padded = [
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            [[1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]]
        self.assertListEqual(padded.tolist(), expected_padded)
