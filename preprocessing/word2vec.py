# Implementation inspired by:
import torch
from torch.autograd import Variable


class Word2Vec:

    def __init__(self, vocab_size, word_idx):
        self.vocab_size = vocab_size
        self.word_idx = word_idx
        self._input_layer = self._get_input_layer()
        self._hidden_layer =
        self.embedding_dims = 5

    def _get_input_layer(self):
        x = torch.zeros(self.vocab_size).float()
        x[self.word_idx] = 1.0
        return x

    def _get_hidden_layer(self):

        W1 = Variable(torch.randn(self.embedding_dims, self.vocab_size).float(), requires_grad=True)
        return torch.matmul(W1, x)


    def word2vec(corpus:[[str]], vocabulary:[str]):
