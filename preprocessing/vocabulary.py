import csv


def create_vocabulary(functions_list: [[]]):
    """
    Creates an vocabulary from the entire dataset after tokenizing and generalizing

    :param functions_list: List of all tokenized functions
    :return: Dictionary of words to their index and vice versa
    """
    vocabulary = []
    for function in functions_list:
        for token in function:
            if token not in vocabulary: vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    return word2idx, idx2word


def write_to_file(dictionary: dict):
    """
    Writes a dictionary to file

    :param dictionary: Dictionary containing words and their corresponding index
    """
    csv_file = "word2vec.csv"
    with open(csv_file, 'w') as file:
        writer = csv.writer(file)
        for key, value in dictionary.items():
            writer.writerow([key, value])
