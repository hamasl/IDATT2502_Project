def get_x_table(x, similarity_table, word2idx):
    """
    Converts each token in x to a word2vec vector, by gathering the vectors from the similarity table.
    :param x: A 2d list containing all the tokens from the functions to be used as data.
    :param similarity_table: A table (2d list) consisting of entries which measures the similarity between the word at its row index and column index.
    :param word2idx: A dictionary which maps a word to its corresponding index.
    :return: The new x where each entry has been converted.
    """
    new_x = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[i])):
            row.append(similarity_table[word2idx[x[i][j]]])
        new_x.append(row)
    return new_x
