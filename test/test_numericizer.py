import unittest
import preprocessing.numericizer as num

class NumericizerTest(unittest.TestCase):

	def test_numericizer(self):
		input = [["hei", "id0", "jeg", "heter", "id1"], ["hei", "her", "er", "en", "test"]]
		expected = [[0, -1, 1, 2, -2], [0, 3, 4, 5, 6]]
		numericized, min_id_val, max_val = num.convert_to_numerical_values(input)
		self.assertEqual(numericized, expected)
		self.assertEqual(min_id_val, -2)
		self.assertEqual(max_val, 6)
