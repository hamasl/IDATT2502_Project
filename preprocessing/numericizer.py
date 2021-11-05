def convert_to_numerical_values(tokens: [[]]) -> ([[]], int, int):
    """
    Converts given tokens to numerical values. If a token is an id,
    the function will convert it into a negative value, and keep track of
    smallest id value. If a token is not an id, the function
    will convert it into a positive value instead. The function also keeps
    track of the biggest positive value.

    :param tokens: 2D array of tokens
    :return: 2D array with numerical values, smallest id value, max positive value
    """
    min_id_val = -1
    dictionary = []
    num_values = tokens.copy()
    for (i, function) in enumerate(tokens):
        for (j, token) in enumerate(function):
            if token[0:2:1] == "id":
                val = -(int(token[2:]) + 1)
                num_values[i][j] = val
                if min_id_val > val:
                    min_id_val = val
            else:
                if token not in dictionary:
                    dictionary.append(token)
                num_values[i][j] = dictionary.index(token)
    max_value = len(dictionary) - 1
    return num_values, min_id_val, max_value
