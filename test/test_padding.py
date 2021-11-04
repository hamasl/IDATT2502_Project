import unittest
import preprocessing.padding as padding

class PaddingTest(unittest.TestCase):
	
	def test_add_padding_multiple_of_4(self):
		input = [[1,2,3], [1,2,3,4], [1]]
		padded, masked = padding.masking_padding(input)
		expected_padded = [[1,2,3,0], [1,2,3,4], [1,0,0,0]]
		expected_masked = [[True, True, True, False], [True, True, True, True], [True, False, False, False]]
		self.assertListEqual(padded.tolist(), expected_padded)
		self.assertListEqual(masked.tolist(), expected_masked)
	

	def test_add_padding_not_multiple_of_4(self):
		input = [[1,2,3,4], [1,2,3,4,5,6], [1,2,3]]
		padded, masked = padding.masking_padding(input)
		expected_padded = [[1,2,3,4,0,0,0,0], [1,2,3,4,5,6,0,0], [1,2,3,0,0,0,0,0]]
		expected_masked = [
			[True, True, True, True, False, False, False, False], 
			[True, True, True, True, True, True, False, False], 
			[True, True, True, False, False, False, False, False]]
		self.assertListEqual(padded.tolist(), expected_padded)
		self.assertListEqual(masked.tolist(), expected_masked)