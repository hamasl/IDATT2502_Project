from io import BytesIO
import tokenize as tn
import re

"""
    Input: File/filename to be tokenized

    Reads each line in file, and removes comments, new_line characters, and c compiler directives.

    Thereafter, a python tokenizer, to tokenizer the array

    Output: Tokenized array of the file
"""

def get_functions(filename):
    functions = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            brackets = 0
            function = ""
            match = re.search("^(unsigned|signed|static)?\s*(void|int|char|short|long|float|double)\s+(\w+)\([^)]*\)\s+{", line)
            if(match): 
                brackets += 1
                function += line

                if '}' in line:
                    brackets = 0


                while brackets > 0:
                    line = f.readline()
                    if '{' in line: brackets += 1
                    if '}' in line: brackets -= 1
                    function += line

                functions.append(function)
            line = f.readline()
        return functions

def tokenize(function_array):
    tokenized_functions = []
    for function in function_array:
        text = []
        is_in_comment = False
        for line in function.splitlines():
            if "/*" in line: is_in_comment = True
                
            if '*/' not in line and is_in_comment: continue
                    
            if is_in_comment: 
                line = line[line.index("*/")+2: -1]
                is_in_comment = False

            if line.startswith("#"):
                continue
            line = line.replace("\n", "")

            text.append(line)
        tokenized = []
        for _, tokval, _, _, _ in tn.tokenize(BytesIO(("".join(text)).encode('utf-8')).readline):
            if(tokval == '' or tokval == 'utf-8' or tokval == ' '): continue
            tokenized.append(tokval)
        
        tokenized_functions.append(tokenized)
        
    return tokenized_functions

if __name__ == '__main__':
    # print(tokenize("CWE835_Infinite_Loop__while_01.c"))
    functions = get_functions("./formatted/CWE835_Infinite_Loop__while_01.c")
    for tokenized in tokenize(functions):
        print(tokenized)
