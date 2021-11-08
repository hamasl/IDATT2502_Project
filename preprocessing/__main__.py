import os

import torch

from preprocessing.tokenizer import tokenize
import preprocessing.keyword_dictionary as keyword_dictionary
import preprocessing.generalizer as generalizer
import preprocessing.numericizer as numericizer
import preprocessing.normalizer as normalizer
import preprocessing.padding as padding

if __name__ == '__main__':
    x, y = tokenize()
    dictionary = keyword_dictionary.get()
    x = generalizer.handle_functions_and_variables(generalizer.handle_literals(x, dictionary), dictionary)
    x, min_val, max_val = numericizer.convert_to_numerical_values(x)
    x = normalizer.normalize(x, min_val, max_val)
    pad = padding.pad(x)
    x = torch.Tensor(pad).reshape(len(pad), 1, len(pad[0]))
    y = torch.LongTensor(y).reshape(len(y))
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    torch.save(x, os.path.join(dirname,"x.pt"))
    torch.save(y, os.path.join(dirname,"y.pt"))

