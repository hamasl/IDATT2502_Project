import os
import sys
from io import BytesIO
import tokenize as tn
import re


class Tokenizer:
    def __init__(self, number_of_types: int):
        self.number_of_types = number_of_types
        self.number_of_good_functions = 0

    def get_functions(self, filename: str, function_array_len: int, class_number: int, num_of_functions: int = sys.maxsize, ignore_main: bool = True):
        """
        Uses regex, to find the functions start, and runs a while loop to find the closing curly bracket
        Only parses one bad function and one good function before it returns

        :param ignore_main: boolean value to ignore main or not
        :param num_of_functions: The number of functions to gather from a file
        :param filename: name of the C file
        :param function_array_len: length of the current array of tokenized functions
        :param class_number: number of classes
        :return: array of functions in given file
        """
        functions = []
        function_types = []
        function_names = []
        with open(filename, 'r') as f:
            line = f.readline()
            # Only get a certain number of functions from each file
            while line and len(functions) < num_of_functions:
                brackets = 0
                function = ""
                match = re.search(
                    "^\s*(unsigned|signed|static)?\s*(void|int|char|short|long|float|double)\s+(\w+)\([^)]*\)\s+{",
                    line)
                if match and ("main" not in line or not ignore_main):
                    function_names.append(line[:-3])
                    if "bad" in line:
                        function_types.append(class_number)
                    # Get only 1/NUMBER_OF_TYPES good functions
                    elif function_array_len % self.number_of_types == 0:
                        self.number_of_good_functions += 1
                        function_types.append(0)
                    else:
                        line = f.readline()
                        continue
                    brackets += 1
                    function += line

                    if '}' in line:
                        brackets = 0

                    while brackets > 0:
                        line = f.readline()
                        if '{' in line: brackets += 1
                        if '}' in line: brackets -= 1
                        function += line

                    functions.append(function)
                line = f.readline()
            if len(functions) != len(function_types): raise Exception("Number of functions not equal number of types")
            return functions, function_types, function_names

    def file_tokenize(self, function_array: [[]]):
        """
        Removes single line comment, multilinecomments, compiler directives and new lines
        It then uses a python tokenizer(https://docs.python.org/3/library/tokenize.html),
        to split character in the function into an array

        :param function_array: array of functions strings
        :return: an array of arrays, each array containing tokenized functions
        """
        tokenized_functions = []
        for function in function_array:
            text = []
            is_in_comment = False
            for line in function.splitlines():
                if "/*" in line:
                    is_in_comment = True

                if '*/' not in line and is_in_comment: continue

                if is_in_comment:
                    line = line[line.index("*/") + 2: -1]
                    is_in_comment = False

                if '//' in line: line = line[:line.index("//")]

                if line.startswith("#"): continue

                line = line.replace("\n", "")

                text.append(line)
            tokenized = []
            for _, tokval, _, _, _ in tn.tokenize(BytesIO(("".join(text)).encode('utf-8')).readline):
                if tokval == '' or tokval == 'utf-8' or tokval == ' ': continue
                tokenized.append(tokval)

            tokenized_functions.append(tokenized)

        return tokenized_functions

    def tokenize(self):
        """
        Runs through every C file an formatted folder.
        Reading two functions from each file and tokenize them into array
        x is a double nested array, where each array in the array is a tokenized function
        y is the type of the function from the nested array where the indexes are the same

        :return: x and y
        """
        print("Tokenizing...")
        x = []
        y = []
        dirname = os.path.join(os.path.dirname(__file__), "../formatted/")
        for index, folder in enumerate(os.listdir(os.path.join(dirname))):
            for file in os.listdir(os.path.join(dirname, folder)):
                functions, types, _ = self.get_functions(os.path.join(dirname, folder, file), len(y), index+1, num_of_functions=2)
                y += types
                for tokenized in self.file_tokenize(functions):
                    x.append(tokenized)
        print(f"Number of function types: {len(y)}")
        return x, y
