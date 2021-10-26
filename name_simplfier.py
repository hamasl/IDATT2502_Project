

def simplify(input, period_length):
    variable_base_name = "ID"
    function_base_name = "FUNC"
    variable_names = []
    function_names = []
    output = []
    #TODO dict also needs to contain BASE tokens for strings and integers
    #TODO get dict from same source as lexer
    dict = ["int", "*", ";", "(", ")", "=", "void", ",", "BASE_INT"]

    for i in range(len(input)):
        if i % period_length == 0:
            variable_names = []
            function_names = []
        if input[i] not in dict:
            if len(input) >= i and input[i+1] == "(":
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
    test = ["int", "a", "=", "BASE_INT", ";", "a", "=", "BASE_INT", ";", "int", "b", "=", "BASE_INT", ";", "void",
            "parse", "(", "int", "c", ",", "int", "d", ")", ";",
            "parse", "(", "a", ",", "b", ")", ";"]
    print(simplify(test, 11))