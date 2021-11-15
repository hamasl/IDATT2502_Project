from preprocessing.word2vec import Word2Vec


def get_similarity_table(num_of_words, model: Word2Vec) -> [[int]]:
    """
    Creates a table consisting where each row and column represents a word.
    An entry containing the similarity between the word at its row index and column index.
    :param num_of_words: The number of words to base the table on. Table dimension is num_of_words X num_of_words.
    :param model: The trained word2vec model to gather the similarities from.
    :return: Returns the table
    """
    word_similarities = []
    for i in range(num_of_words):
        row = []
        for j in range(num_of_words):
            if i == j:
                row.append(1)
            else:
                row.append(model.similarity(i, j).item())
        word_similarities.append(row)
    return word_similarities
