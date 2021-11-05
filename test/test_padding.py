import unittest
import preprocessing.padding as padding

class PaddingTest(unittest.TestCase):
	
	def test_add_padding_multiple_of_4(self):
		input = [[1,2,3], [1,2,3,4], [1]]
		padded = padding.pad(input)
		expected_padded = [[1,2,3,0], [1,2,3,4], [1,0,0,0]]
		self.assertListEqual(padded.tolist(), expected_padded)


	def test_add_padding_not_multiple_of_4(self):
		input = [[1,2,3,4], [1,2,3,4,5,6], [1,2,3]]
		padded = padding.pad(input)
		expected_padded = [[1,2,3,4,0,0,0,0], [1,2,3,4,5,6,0,0], [1,2,3,0,0,0,0,0]]
		self.assertListEqual(padded.tolist(), expected_padded)