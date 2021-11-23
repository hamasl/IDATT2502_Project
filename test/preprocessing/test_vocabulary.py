import unittest
import preprocessing.vocabulary as vocabulary


class VocabularyTest(unittest.TestCase):

    def test_create_vocabulary(self):
        input_list = [['abc', '123', 'test'], ['abc', '231']]
        word2idx = vocabulary.create_vocabulary(input_list)

        self.assertEqual(word2idx['123'], 1)
