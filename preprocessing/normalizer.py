def normalize(inp: [[]], min_id_val: int, max_val: int) -> [[]]:
    """
    Normalizes the values of the 2D token array given as
    parameter. Uses given smallest id value and biggest
    token value, thus outputting values from 0 to 1.

    :param inp: input as 2D array of tokens
    :param min_id_val: smallest id value
    :param max_val: biggest positive token value
    :return: 2D normalized token array
    """
    normalized = inp.copy()
    for (i, function) in enumerate(inp):
        for (j, token) in enumerate(function):
            normalized[i][j] = (token - (min_id_val - 1)) / (max_val - (min_id_val - 1))
    return normalized
