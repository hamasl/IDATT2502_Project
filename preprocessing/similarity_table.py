from preprocessing.word2vec import Word2Vec


def get_similarity_table(num_of_words, model: Word2Vec):
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
