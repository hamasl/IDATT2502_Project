import unittest
import name_simplifier as ns


class NameSimplifierTest(unittest.TestCase):

    def test_convert_to_base_types_str(self):
        input = ["char", "*", "a", "=", '"hei hei"']
        expected = ["char", "*", "a", "=", 'BASE_STRING']
        self.assertEqual(ns.convert_to_base_types(input), expected)

    #TODO improve ttest
    def test_convert_to_base_types_str(self):
        input = ["char", "*", "a", "=", '"hei hei"']
        expected = ["char", "*", "a", "=", 'BASE_STRING']
        self.assertEqual(ns.convert_to_base_types(input), expected)
