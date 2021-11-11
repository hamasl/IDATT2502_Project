def get_x_table(x, similarity_table, word2idx):
    new_x = []
    for i in range(len(x)):
        row = []
        for j in range(len(x[i])):
            row.append(similarity_table[word2idx[x[i][j]]])
        new_x.append(row)
    return new_x
