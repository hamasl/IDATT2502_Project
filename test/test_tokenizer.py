import unittest
import preprocessing.tokenizer as tn
import os

class TokenizerTest(unittest.TestCase):

	def test_tokenize_lines(self):
		input = ["int i = 0;\n if(1) return 1;\n"]
		expected = [['int', 'i', '=', '0', ';', 'if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.file_tokenize(input), expected)

	def test_remove_comments(self):
		input = ["/*int i = 0;*/\n if(1) return 1;\n"]
		expected = [['if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.file_tokenize(input), expected)

	def test_remove_compiler_directives(self):
		input = ["#include 'stdio'\n#DEFINE ABC 123\n if(1) return 1;\n"]
		expected = [['if', '(', '1', ')', 'return', '1', ';']]
		self.assertEqual(tn.file_tokenize(input), expected)

	def test_read_functions_from_file(self):
		dirname = os.path.join(os.path.dirname(__file__))
		functions, types = tn.get_functions(os.path.join(dirname,"test_data/CWE835_Infinite_Loop__for_01.c"), 1, 0)
		self.assertEqual(functions[0], 'void CWE835_Infinite_Loop__for_01_bad() {\n  int i = 0;\n\n  /* FLAW: Infinite Loop - for() with no break point */\n  for (i = 0; i >= 0; i = (i + 1) % 256) {\n    printIntLine(i);\n  }\n}\n')
		self.assertEqual(functions[1], 'static void good1() {\n  int i = 0;\n\n  for (i = 0; i >= 0; i = (i + 1) % 256) {\n    /* FIX: Add a break point for the loop if i = 10 */\n    if (i == 10) {\n      break;\n    }\n    printIntLine(i);\n  }\n}\n')	
		self.assertListEqual(types, [1, 0])
