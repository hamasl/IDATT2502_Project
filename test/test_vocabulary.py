import unittest
import preprocessing.vocabulary as vocabulary

class VocabularTesty(unittest.TestCase):

	def test_create_vocabulary(self):
		input_list = [['abc', '123', 'test'], ['abc', '231']]
		word2idx, idx2word = vocabulary.create_vocabulary(input_list)

		self.assertEqual(idx2word[0], 'abc')
		self.assertEqual(word2idx['123'], 1)
	

	def test_create_vocabulary_first_list_empty(self):
		input_list = [[], ['abc', '231']]
		word2idx, idx2word = vocabulary.create_vocabulary(input_list)

		self.assertEqual(idx2word[0], 'abc')
		self.assertEqual(word2idx['231'], 1)
	


	