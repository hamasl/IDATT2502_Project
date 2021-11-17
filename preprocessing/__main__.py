import os

import torch

import preprocessing.tokenizer as tokenizer
import preprocessing.keyword_dictionary as keyword_dictionary
import preprocessing.generalizer as generalizer
import preprocessing.vocabulary as vocab
import preprocessing.vocabulary_pairing as vocab_pairing
import preprocessing.word2vec as word2vec
from preprocessing.similarity_table import get_similarity_table
from preprocessing.padding import pad
from preprocessing.x_table import get_x_table

def process(file_path):
    

if __name__ == '__main__':
    tkn = tokenizer.Tokenizer(13)
    x, y = tkn.tokenize()
    dictionary = keyword_dictionary.get_keywords()
    x = generalizer.handle_functions_and_variables(generalizer.handle_literals(x, dictionary), dictionary)
    word2idx, idx2word = vocab.create_vocabulary(x)
    index_pairing = vocab_pairing.index_pairing(x, word2idx)
    device = torch.device("cpu")
    print(f"Running on: {device}")
    word2vec_model = word2vec.Word2Vec(len(word2idx), word2idx, device)
    print("Training word2vec model:")
    word2vec_model.train(1, 0.015, index_pairing, verbose=True)
    print("Completed word2vec training")
    similarity_table = get_similarity_table(len(word2idx), word2vec_model)
    x = get_x_table(x, similarity_table, word2idx)
    x = pad(x, len(word2idx))
    x_shape = x.shape
    x = torch.reshape(x, (x_shape[0], 1, x_shape[1], x_shape[2]))
    y = torch.LongTensor(y).reshape(len(y))
    dirname = os.path.join(os.path.dirname(__file__), "../processed")
    torch.save(x, os.path.join(dirname, "x.pt"))
    torch.save(y, os.path.join(dirname, "y.pt"))
