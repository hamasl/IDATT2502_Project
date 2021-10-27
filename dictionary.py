import csv
from os.path import isfile


def addToDictionary(inp, out):
    dictionary = {}
    index = 0
    if isfile(out):
        with open(out, mode='r') as infile:
            reader = csv.reader(infile)
            dictionary = {rows[0]: rows[1] for rows in reader}
            index = int(list(dictionary.keys())[-1]) + 1
    for token in inp:
        if token not in dictionary.values():
            dictionary[index] = token
            index += 1
    with open(out, mode='w') as outfile:
        writer = csv.writer(outfile)
        for key, value in dictionary.items():
            writer.writerow([key, value])
