import preprocessing.simplifier_constants as sc


def handle_literals(inp: [[str]], dictionary: [str]) -> [[str]]:
    """
    Converts literals of type float, char and str to BASE_FLOAT, BASE_CHAR and BASE_STRING respectively.
    Literals of type int are split into digits.
    :param inp: A 2d list where each row is a tokenized function.
    :param dictionary: The dictionary of keywords to avoid generalizing.
    :return: A copy of input but with converted literals.
    """
    output = []
    for i in range(len(inp)):
        row = []
        for j in range(len(inp[i])):
            if inp[i][j] not in dictionary:
                if inp[i][j].startswith('"') and inp[i][j].endswith('"'):
                    row.append(sc.BASE_STRING)
                elif inp[i][j].startswith("'") and inp[i][j].endswith("'"):
                    row.append(sc.BASE_CHAR)
                elif "." in inp[i][j] and inp[i][j].replace(".", "", 1).isdigit():
                    row.append(sc.BASE_FLOAT)
                else:
                    row.extend(list(inp[i][j]))
            else:
                row.append(inp[i][j])
        output.append(row)
    return output


# TODO add sc.BASE_CHAR and so on to dictionary
def handle_functions_and_variables(inp: [[str]], dictionary: [str]) -> [[str]]:
    """
    Converts function names to FUNC. Variable names are changed to ID0, ID1 ... When the same variable name is hit it is give the same ID.
    When a new row is used the variable names are reset, and we start again at ID0, ID1 ...
    :param inp: A 2d list where each row is a tokenized function.
    :param dictionary: The dictionary of keywords to avoid generalizing.
    :return: A copy of input but with each  converted literals.
    """
    output = []
    for i in range(len(inp)):
        row = []
        variable_names = []
        for j in range(len(inp[i])):
            if inp[i][j] not in dictionary:
                if len(inp[i]) > j + 1 and inp[i][j + 1] == "(":
                    row.append(sc.GENERIC_FUNCTION_NAME)
                elif not inp[i][j].isdigit():
                    if inp[i][j] in variable_names:
                        row.append(sc.GENERIC_VARIABLE_BASE_NAME + str(variable_names.index(inp[i][j])))
                    else:
                        row.append(sc.GENERIC_VARIABLE_BASE_NAME + str(len(variable_names)))
                        variable_names.append(inp[i][j])
                else:
                    row.append(inp[i][j])
            else:
                row.append(inp[i][j])
        output.append(row)
    return output
