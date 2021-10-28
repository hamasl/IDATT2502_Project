import unittest
import tokenizer as tn

class TokenizerTest(unittest.TestCase):

	def test_tokenize_lines(self):
		input = ["int i = 0;\n if(1) return 1;\n"]
		expected = [['int', 'i', '=', '0', ';', 'if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.tokenize(input), expected)

	def test_remove_comments(self):
		input = ["/*int i = 0;*/\n if(1) return 1;\n"]
		expected = [['if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.tokenize(input), expected)

	def test_remove_compiler_directives(self):
		input = ["#include 'stdio'\n#DEFINE ABC 123\n if(1) return 1;\n"]
		expected = [['if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.tokenize(input), expected)