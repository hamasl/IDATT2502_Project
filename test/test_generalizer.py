import unittest
import preprocessing.generalizer as ns
import preprocessing.generalizer_constants as sc
import preprocessing.keyword_dictionary as kd


class NameSimplifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dictionary = kd.get()

    def test_handle_literals_str(self):
        inp = [["char", "*", "a", "=", '"hei hei"']]
        expected = [["char", "*", "a", "=", sc.BASE_STRING]]
        self.assertEqual(ns.handle_literals(inp, self.dictionary), expected)

    def test_handle_literals_char(self):
        inp = [["char", "a", "=", "'c'"]]
        expected = [["char", "a", "=", sc.BASE_CHAR]]
        self.assertEqual(ns.handle_literals(inp, self.dictionary), expected)

    def test_handle_literals_float(self):
        inp = [["float", "a", "=", "1.343"]]
        expected = [["float", "a", "=", sc.BASE_FLOAT]]
        self.assertEqual(ns.handle_literals(inp, self.dictionary), expected)

    def test_handle_literals_int(self):
        inp = [["int", "a", "=", "132"]]
        expected = [["int", "a", "=", "1", "3", "2"]]
        self.assertEqual(ns.handle_literals(inp, self.dictionary), expected)

    def test_handle_functions_and_variables_int(self):
        inp = [["1", "3", "2"]]
        expected = [["1", "3", "2"]]
        self.assertEqual(ns.handle_functions_and_variables(inp, self.dictionary), expected)

    def test_handle_functions_and_variables_multiple_variables_in_same_func(self):
        inp = [
            ["float", "a", "=", sc.BASE_CHAR, ";", "float", "b", "=", sc.BASE_CHAR, ";", "a", "=", sc.BASE_CHAR, ";"]]
        expected = [["float", sc.GENERIC_VARIABLE_BASE_NAME + "0", "=", sc.BASE_CHAR, ";", "float",
                     sc.GENERIC_VARIABLE_BASE_NAME + "1", "=", sc.BASE_CHAR, ";", sc.GENERIC_VARIABLE_BASE_NAME + "0",
                     "=", sc.BASE_CHAR, ";"]]
        self.assertEqual(ns.handle_functions_and_variables(inp, self.dictionary), expected)

    def test_handle_functions_and_variables_multiple_functions_ID_resets(self):
        inp = [["float", "a", "=", sc.BASE_CHAR, ";"], ["float", "b", "=", sc.BASE_CHAR, ";"]]
        expected = [["float", sc.GENERIC_VARIABLE_BASE_NAME + "0", "=", sc.BASE_CHAR, ";"], ["float",
                                                                                             sc.GENERIC_VARIABLE_BASE_NAME + "0",
                                                                                             "=", sc.BASE_CHAR, ";"]]
        self.assertEqual(ns.handle_functions_and_variables(inp, self.dictionary), expected)

    def test_handle_functions_and_variables_func_name(self):
        inp = [["void", "parse", "(", "int", "a", ")", "{", "a", "+", "+", ";", "}"]]
        expected = [["void", sc.GENERIC_FUNCTION_NAME, "(", "int", sc.GENERIC_VARIABLE_BASE_NAME + "0", ")", "{",
                     sc.GENERIC_VARIABLE_BASE_NAME + "0", "+", "+", ";", "}"]]
        self.assertEqual(ns.handle_functions_and_variables(inp, self.dictionary), expected)
