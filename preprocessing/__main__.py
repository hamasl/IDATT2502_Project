import os

import torch

from preprocessing.tokenizer import tokenize
import preprocessing.keyword_dictionary as keyword_dictionary
import preprocessing.generalizer as generalizer
import preprocessing.vocabulary as vocab
import preprocessing.vocabulary_pairing as vocab_pairing
import preprocessing.word2vec as word2vec


if __name__ == '__main__':
    x, y = tokenize()
    dictionary = keyword_dictionary.get()
    x = generalizer.handle_functions_and_variables(generalizer.handle_literals(x, dictionary), dictionary)
    word2idx, idx2word = vocab.create_vocabulary(x)
    x = vocab_pairing.index_pairing(x, word2idx)
    print(x)
    print(len(word2idx))
    word2vec_model = word2vec.Word2Vec(len(word2idx), word2idx)
    print("Train model")
    word2vec_model.train(10, 0.0001, x, verbose=True)
    x = word2vec_model.f(x)
    print(x)
    """
    x, min_val, max_val = numericizer.convert_to_numerical_values(x)
    x = normalizer.normalize(x, min_val, max_val)
    pad = padding.pad(x)
    x = torch.Tensor(pad).reshape(len(pad), 1, len(pad[0]))
    y = torch.LongTensor(y).reshape(len(y))
    """
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    torch.save(x, os.path.join(dirname, "x.pt"))
    torch.save(y, os.path.join(dirname, "y.pt"))
