# Implementation inspired by:
import numpy as np
import torch
from torch.autograd import Variable
from torch.autograd.grad_mode import F


class Word2Vec:

    def __init__(self, vocab_size, word_idx):
        self.vocab_size = vocab_size
        self.word_idx = word_idx
        self._input_layer = self._get_input_layer()
        self.embedding_dims = 5

    def _get_input_layer(self):
        x = torch.zeros(self.vocab_size).float()
        x[self.word_idx] = 1.0
        return x

    def train(self, num_epochs, learning_rate, index_pairing):
        W1 = Variable(torch.randn(self.embedding_dims, self.vocab_size).float(), requires_grad=True)
        W2 = Variable(torch.randn(self.vocab_size, self.embedding_dims).float(), requires_grad=True)

        for epo in range(num_epochs):
            loss_val = 0
            for data, target in index_pairing:
                x = Variable(self._input_layer(data)).float()
                y_true = Variable(torch.from_numpy(np.array([target])).long())

                z1 = torch.matmul(W1, x)
                z2 = torch.matmul(W2, z1)

                log_softmax = F.log_softmax(z2, dim=0)

                loss = F.nll_loss(log_softmax.view(1, -1), y_true)
                loss_val += loss.data[0]
                loss.backward()
                W1.data -= learning_rate * W1.grad.data
                W2.data -= learning_rate * W2.grad.data

                W1.grad.data.zero_()
                W2.grad.data.zero_()
            if epo % 10 == 0:
                print(f'Loss at epo {epo}: {loss_val / len(index_pairing)}')