import csv
from os.path import isfile


def readDictionary(dictFile):
    dictionary = {}
    if isfile(dictFile):
        with open(dictFile, mode='r') as infile:
            reader = csv.reader(infile)
            dictionary = {rows[0]: rows[1] for rows in reader}
    return dictionary


def addToDictionary(inp, out):
    index = 0
    dictionary = readDictionary(out)
    if dictionary:
        index = int(list(dictionary.keys())[-1]) + 1
    for token in inp:
        if token not in dictionary.values():
            dictionary[index] = token
            index += 1
    with open(out, mode='w') as outfile:
        writer = csv.writer(outfile)
        for key, value in dictionary.items():
            writer.writerow([key, value])


def convertToNumericalValues(tokens, dictFile):
    num_values = []
    index = 0
    dictionary = readDictionary(dictFile)
    if dictionary:
        index = int(list(dictionary.keys())[-1]) + 1
    key_list = list(dictionary.keys())
    val_list = list(dictionary.values())
    for token in tokens:
        if token in dictionary.values():
            num_values.append(int(key_list[val_list.index(token)]))
        else:
            num_values.append(index)
            with open(dictFile, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([index, token])
                index += 1
    return num_values
