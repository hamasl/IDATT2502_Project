import simplifier_constants as sc

"""

Converts all strings, chars and floats to generic token symboling their type. 
Input: ["int", "a", "=", "33", ";", "a", "=", "35", ";", "int", "b", "=", "40", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
Output: ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
"""

dict = ["+", "}", "{", "float", "char", "int", "*", ";", "(", ")", "=", "void", ",", sc.BASE_STRING, sc.BASE_FLOAT,
        sc.BASE_CHAR]


def convert_to_base_types(input):
    output = []
    for i in range(len(input)):
        if input[i] not in dict:
            if input[i].startswith('"') and input[i].endswith('"'):
                output.append(sc.BASE_STRING)
            elif input[i].startswith("'") and input[i].endswith("'"):
                output.append(sc.BASE_CHAR)
            elif "." in input[i] and input[i].replace(".", "", 1).isdigit():
                output.append(sc.BASE_FLOAT)
            else:
                output.append(input[i])

        else:
            output.append(input[i])
    return output


"""
Converts function names to FUNC. Variable names are changed to ID0, ID1 ... When the same variable name is hit it is give the same ID.
Period length decides the number of iterations before the variable_names array is emptied and the we start again at ID0, ID1 ...
This is done to be able to send multiple functions at the same time.
"""


def simplify(input, period_length):
    variable_names = []
    output = []
    for i in range(len(input)):
        if i % period_length == 0:
            variable_names = []
        if input[i] not in dict:
            if len(input) > i + 1 and input[i + 1] == "(":
                output.append(sc.GENERIC_FUNCTION_NAME)
            elif not input[i].isdigit():
                if input[i] in variable_names:
                    output.append(sc.GENERIC_VARIABLE_BASE_NAME + str(variable_names.index(input[i])))
                else:
                    output.append(sc.GENERIC_VARIABLE_BASE_NAME + str(len(variable_names)))
                    variable_names.append(input[i])
            else:
                output.extend(list(input[i]))
        else:
            output.append(input[i])
    return output
