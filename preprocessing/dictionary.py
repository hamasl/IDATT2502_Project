import csv
from os.path import isfile

max_id = -1


def readDictionary(dictFile: str) -> dict:
    """
    Reads dictionary from a .csv file specified by dictFile parameter.

    :param dictFile: path to dictionary .csv file as string
    :return: dictionary of read tokens
    """
    dictionary = {}
    if isfile(dictFile):
        with open(dictFile, mode='r') as infile:
            reader = csv.reader(infile)
            dictionary = {rows[0]: rows[1] for rows in reader}
    return dictionary


def addToDictionary(inp: [[]], out: str):
    """
    Reads dictionary if it exists (creates a new if it doesn't)
    and adds token values to the dictionary that do not already
    exist in it. Furthermore, if the function specifically encounters
    an id value, it will add it to the dictionary with negative value.

    :param inp: input as 2D array of tokens
    :param out: path to dictionary .csv file as string
    """
    global max_id
    index = 0
    dictionary = readDictionary(out)
    if dictionary:
        index = int(list(dictionary.keys())[-1]) + 1
    for function in inp:
        for token in function:
            if token[0:2:1] == "id" and max_id > (-int(token[2:]) - 1):
                max_id = -int(token[2:]) - 1
            if token not in dictionary.values() and token[0:2:1] != "id":
                dictionary[index] = token
                index += 1
    with open(out, mode='w') as outfile:
        writer = csv.writer(outfile)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def convertToNumericalValues(tokens: [[]], dictFile: str) -> [[]]:
    """
    Reads dictionary if it exists and converts tokens into numerical values
    specified in the dictionary. If a token value does not exist in the current
    dictionary, it appends the newly encountered value to the dictionary, and gives
    it a new numerical value. Furthermore, if the function specifically encounters
    an id value, it will not convert it to a positive numerical value, but it will
    convert it to a negative value.

    :param tokens: 2D array of tokens
    :param dictFile: path to dictionary .csv file as string
    :return: 2D array with numerical values
    """
    global max_id
    num_values = []
    index = 0
    dictionary = readDictionary(dictFile)
    if dictionary:
        index = int(list(dictionary.keys())[-1]) + 1
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    for function in tokens:
        for token in function:
            if token[0:2:1] != "id":
                val = -int(token[2:])
                num_values.append(val)
                if max_id > (-int(token[2:]) - 1):
                    max_id = -int(token[2:]) - 1
            if token in dictionary.values():
                num_values.append(int(key_list[val_list.index(token)]))
            else:
                num_values.append(index)
                with open(dictFile, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([index, token])
                    index += 1
    return num_values
