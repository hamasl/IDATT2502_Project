import unittest
import preprocessing.name_simplifier as ns
import preprocessing.simplifier_constants as sc


class NameSimplifierTest(unittest.TestCase):

    def test_convert_to_base_types_str(self):
        input = [["char", "*", "a", "=", '"hei hei"']]
        expected = [["char", "*", "a", "=", sc.BASE_STRING]]
        self.assertEqual(ns.convert_to_base_types(input), expected)

    def test_convert_to_base_types_char(self):
        input = [["char", "*", "a", "=", "'c'"]]
        expected = [["char", "*", "a", "=", sc.BASE_CHAR]]
        self.assertEqual(ns.convert_to_base_types(input), expected)

    def test_convert_to_base_types_float(self):
        input = [["char", "*", "a", "=", "1.343"]]
        expected = [["char", "*", "a", "=", sc.BASE_FLOAT]]
        self.assertEqual(ns.convert_to_base_types(input), expected)

    def test_convert_to_base_types_int(self):
        input = [["char", "*", "a", "=", "132"]]
        expected = [["char", "*", "a", "=", "132"]]
        self.assertEqual(ns.convert_to_base_types(input), expected)

    def test_simplify_int(self):
        input = [["132"]]
        expected = [["1", "3", "2"]]
        self.assertEqual(ns.simplify(input), expected)

    def test_simplify_multiple_variables_in_same_func(self):
        input = [
            ["float", "a", "=", sc.BASE_CHAR, ";", "float", "b", "=", sc.BASE_CHAR, ";", "a", "=", sc.BASE_CHAR, ";"]]
        expected = [["float", sc.GENERIC_VARIABLE_BASE_NAME + "0", "=", sc.BASE_CHAR, ";", "float",
                     sc.GENERIC_VARIABLE_BASE_NAME + "1", "=", sc.BASE_CHAR, ";", sc.GENERIC_VARIABLE_BASE_NAME + "0",
                     "=", sc.BASE_CHAR, ";"]]
        self.assertEqual(ns.simplify(input), expected)

    def test_simplify_multiple_functions_ID_resets(self):
        input = [["float", "a", "=", sc.BASE_CHAR, ";"], ["float", "b", "=", sc.BASE_CHAR, ";"]]
        expected = [["float", sc.GENERIC_VARIABLE_BASE_NAME + "0", "=", sc.BASE_CHAR, ";"], ["float",
                     sc.GENERIC_VARIABLE_BASE_NAME + "0", "=", sc.BASE_CHAR, ";"]]
        self.assertEqual(ns.simplify(input), expected)

    def test_simplify_func_name(self):
        input = [["void", "parse", "(", "int", "a", ")", "{", "a", "+", "+", ";", "}"]]
        expected = [["void", sc.GENERIC_FUNCTION_NAME, "(", "int", sc.GENERIC_VARIABLE_BASE_NAME + "0", ")", "{",
                     sc.GENERIC_VARIABLE_BASE_NAME + "0", "+", "+", ";", "}"]]
        self.assertEqual(ns.simplify(input), expected)
