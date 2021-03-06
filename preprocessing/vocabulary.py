import csv
import os.path


def create_vocabulary(functions_list: [[]]):
    """
    Creates an vocabulary from the entire dataset after tokenizing and generalizing

    :param functions_list: List of all tokenized functions
    :return: Dictionary of words to their index and vice versa
    """
    vocabulary = []
    for function in functions_list:
        for token in function:
            if token not in vocabulary:
                vocabulary.append(token)

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}

    return word2idx, vocabulary


def write_to_file(dictionary: dict):
    """
    Writes a dictionary to file

    :param dictionary: Dictionary containing words and their corresponding index
    """

    csv_file = "state/vocab.csv"
    with open(os.path.join(os.path.dirname(__file__), csv_file), 'w') as file:
        writer = csv.writer(file)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def read_from_file():
    """
    Reads vocabulary from file and returns it as dictionary

    :return: Dictionary
    """
    dictionary = {}
    csv_file = "state/vocab.csv"
    with open(os.path.join(os.path.dirname(__file__), csv_file), 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            dictionary[row[0]] = int(row[1])
    return dictionary
