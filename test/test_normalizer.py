import unittest
import preprocessing.normalizer as nm

class NormalizerTest(unittest.TestCase):

	def test_normalize(self):
		input = [[2, 3, -1, 5, 2], [10, -3, 1, -5, 2]]
		expected = [[1/2, 9/16, 5/16, 11/16, 1/2], [1.0, 3/16, 7/16, 1/16, 1/2]]
		min_id_val = -5
		max_val = 10
		self.assertEqual(nm.normalize(input, min_id_val, max_val), expected)
