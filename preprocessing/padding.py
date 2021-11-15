import torch


def pad(inp, num_of_words):
    """
    Finds the largest tokenized function array.
    Makes each tokenized array the size of this number, and pads with 0 to create a padded version of the tokenized array
    :param inp: A nested array containing tokenized functions
    :param num_of_words:
    :return: padded array and masked array
    """
    max_len_row = len(max(inp, key=len))
    # Padded length must be multiple of 4
    max_len_row += (max_len_row % 4)
    padded = torch.zeros((len(inp), max_len_row, num_of_words))

    for i in range(len(inp)):
        current_row_length = len(inp[i])
        padded[i, 0:current_row_length] = torch.tensor(inp[i])

    return padded
