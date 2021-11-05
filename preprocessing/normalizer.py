import preprocessing.dictionary as dc


def normalize(inp: [[]], d: dict) -> [[]]:
    """
    Normalizes the values of the 2D token array given as
    parameter. Uses given dictionary's length to normalize
    the values, thus outputting values from 0 to 1.

    :param inp: input as 2D array of tokens
    :param d: dictionary
    :return: 2D normalized token array
    """
    normalized = inp.copy()
    size = len(d)
    for (i, function) in enumerate(inp):
        for (j, token) in enumerate(function):
            normalized[i][j] = (token - (dc.max_id - 1)) / (size - (dc.max_id - 1))
    return normalized

