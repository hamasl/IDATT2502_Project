from io import BytesIO
import tokenize as tn

"""
    Input: File/filename to be tokenized

    Reads each line in file, and removes comments, new_line characters, and c compiler directives.

    Thereafter, a python tokenizer, to tokenizer the array

    Output: Tokenized array of the file
"""
def tokenize(filename):
    text = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if "/*" in line:
                while "*/" not in line:
                    line = f.readline()
                line = line[line.index("*/")+2: -1]

            if line.startswith("#"):
                line = f.readline()

            line = line.replace("\n", "")

            text.append(line)
            line = f.readline()

    tokenized_gen = tn.tokenize(BytesIO(("".join(text)).encode('utf-8')).readline)

    tokenized = []
    for _, tokval, _, _, _ in tokenized_gen:
        if(tokval == '' or tokval == 'utf-8'): continue
        tokenized.append(tokval)

    return tokenized

if __name__ == '__main__':
    print(tokenize("CWE835_Infinite_Loop__while_01.c"))
