import preprocessing.simplifier_constants as sc

"""

Converts all strings, chars and floats to generic token symboling their type. 
Input: [["int", "a", "=", "33", ";", "a", "=", "35", ";", "int", "b", "=", "40", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]]
Output: [["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]]
"""

dict = ["+", "}", "{", "float", "char", "int", "*", ";", "(", ")", "=", "void", ",", sc.BASE_STRING, sc.BASE_FLOAT,
        sc.BASE_CHAR]


def convert_to_base_types(input):
    output = []
    for i in range(len(input)):
        row = []
        for j in range(len(input[i])):
            if input[i][j] not in dict:
                if input[i][j].startswith('"') and input[i][j].endswith('"'):
                    row.append(sc.BASE_STRING)
                elif input[i][j].startswith("'") and input[i][j].endswith("'"):
                    row.append(sc.BASE_CHAR)
                elif "." in input[i][j] and input[i][j].replace(".", "", 1).isdigit():
                    row.append(sc.BASE_FLOAT)
                else:
                    row.append(input[i][j])
            else:
                row.append(input[i][j])
        output.append(row)
    return output


"""
Converts function names to FUNC. Variable names are changed to ID0, ID1 ... When the same variable name is hit it is give the same ID.
When a new row is used the variable names are reset, and we start again at ID0, ID1 ...
"""


def simplify(input):
    output = []
    for i in range(len(input)):
        row = []
        variable_names = []
        for j in range(len(input[i])):
            if input[i][j] not in dict:
                if len(input[i]) > j + 1 and input[i][j + 1] == "(":
                    row.append(sc.GENERIC_FUNCTION_NAME)
                elif not input[i][j].isdigit():
                    if input[i][j] in variable_names:
                        row.append(sc.GENERIC_VARIABLE_BASE_NAME + str(variable_names.index(input[i][j])))
                    else:
                        row.append(sc.GENERIC_VARIABLE_BASE_NAME + str(len(variable_names)))
                        variable_names.append(input[i][j])
                else:
                    row.extend(list(input[i][j]))
            else:
                row.append(input[i][j])
        output.append(row)
    return output
