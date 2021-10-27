"""
Input: ["int", "a", "=", "33", ";", "a", "=", "35", ";", "int", "b", "=", "40", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
Output: ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
"""

dict = ["int", "*", ";", "(", ")", "=", "void", ",", "BASE_INT"]


def convert_to_base_types(input):
    output = []
    for i in range(len(input)):
        if input[i] not in dict:
            if input[i].startswith('"') and input[i].endswith('"'):
                output.append("BASE_STRING")
            elif input[i].startswith("'") and input[i].endswith("'"):
                output.append("BASE_CHAR")
            elif "." in input[i] and input[i].replace(".", "", 1).isdigit():
                output.append("BASE_FLOAT")
            else:
                output.append(input[i])

        else:
            output.append(input[i])
    return output


"""
Input: ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "a", ",", "b", ")", ";"]
Output: [
    'FUNC0', '(', 'BASE_INT', ',', 'BASE_INT', ')', ';']
"""


# int a = [10]; a[12] = input()
# int a[BASE_INT]; a [BASE_INT]
# int a  = 2; a += 2; printf(a)
# printf(4)
def simplify(input, period_length):
    function_base_name = "FUNC"
    variable_base_name = "ID"
    variable_names = []
    output = []
    # TODO dict also needs to contain BASE tokens for strings and integers
    # TODO get dict from same source as lexer

    for i in range(len(input)):
        if i % period_length == 0:
            variable_names = []
        if input[i] not in dict:
            if len(input) >= i and input[i + 1] == "(":
                output.append(function_base_name)
            elif not input[i].isdigit():
                if input[i] in variable_names:
                    output.append(variable_base_name + str(variable_names.index(input[i])))
                else:
                    output.append(variable_base_name + str(len(variable_names)))
                    variable_names.append(input[i])
            else:
                output.extend(list(input[i]))
        else:
            output.append(input[i])
    return output

"""
Input: ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
Output: ['int', 'ID0', '=', 'BASE_INT', ';', 'ID0', '=', 'BASE_INT', ';', 'int', 'ID1', '=', 'BASE_INT', ';',
            'void', 'FUNC0', '(', 'int', 'ID0', ',', 'int', 'ID1', ')', ';',
            'FUNC0', '(', 'ID0', ',', 'ID1', ')', ';']
"""


def simplify_old(input, period_length):
    variable_base_name = "ID"
    function_base_name = "FUNC"
    variable_names = []
    function_names = []
    output = []
    # TODO dict also needs to contain BASE tokens for strings and integers
    # TODO get dict from same source as lexer

    for i in range(len(input)):
        if i % period_length == 0:
            variable_names = []
            function_names = []
        if input[i] not in dict:
            if len(input) >= i and input[i + 1] == "(":
                if input[i] in function_names:
                    output.append(function_base_name + str(function_names.index(input[i])))
                else:
                    output.append(function_base_name + str(len(function_names)))
                    function_names.append(input[i])
            else:
                if input[i] in variable_names:
                    output.append(variable_base_name + str(variable_names.index(input[i])))
                else:
                    output.append(variable_base_name + str(len(variable_names)))
                    variable_names.append(input[i])
        else:
            output.append(input[i])
    return output


if __name__ == '__main__':
    test1 = ["int", "a", "=", "33", ";", "a", "=", "35", ";", "int", "b", "=", "40", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
    print(test1)
    print(simplify(convert_to_base_types(test1),150))

    test = ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
    #print(simplify(test, 150))
