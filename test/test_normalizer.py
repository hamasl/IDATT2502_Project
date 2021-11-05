import unittest
import preprocessing.normalizer as nm
import preprocessing.dictionary as dc

class NormalizerTest(unittest.TestCase):

	def test_normalize(self):
		test_inp = [["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "id0", "id1", "id2", "id3", "id4"]]
		dc.add_to_dictionary(test_inp, "test_dict.csv")
		dictionary = dc.read_dictionary("test_dict.csv")
		input = [[2, 3, -1, 5, 2], [10, -3, 1, -5, 2]]
		expected = [[1/2, 9/16, 5/16, 11/16, 1/2], [1.0, 3/16, 7/16, 1/16, 1/2]]
		self.assertEqual(dc.max_id, -5)
		self.assertEqual(nm.normalize(input, dictionary), expected)
