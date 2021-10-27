from io import BytesIO
from tokenize import tokenize
text = []

with open("data/CWE835_Infinite_Loop__while_01.c", 'r') as f:
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

test = tokenize(BytesIO(("".join(text)).encode('utf-8')).readline)

tokenized = []
for toknum, tokval, _, _, _ in test:
	tokenized.append(tokval)

print(tokenized)
