# Implementation inspired by: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
import torch
from torch.nn.functional import log_softmax, nll_loss


class Word2Vec:

    def __init__(self, vocab_size, word_idx, device=torch.device("cpu")):
        self.vocab_size = vocab_size
        self.word_idx = word_idx
        self.embedding_dims = 5
        self._device = device
        self._W1 = torch.randn(self.embedding_dims, self.vocab_size, device=device, requires_grad=True, dtype=torch.float)
        self._W2 = torch.randn(self.vocab_size, self.embedding_dims, device=device, requires_grad=True, dtype=torch.float)

    def get_input_layer(self, word_idx):
        x = torch.zeros(self.vocab_size, device=self._device, dtype=torch.float)
        x[word_idx] = 1.0
        return x

    def f(self, x):
        return torch.matmul(self._W2, torch.matmul(self._W1, x))

    def similarity(self, word_idx1, word_idx2):
        return torch.dot(self._W2[word_idx1], self._W2[word_idx2]) / (torch.norm(self._W2[word_idx1]) * torch.norm(self._W2[word_idx2]))


    def train(self, num_epochs, learning_rate, index_pairing, verbose=False):

        for epo in range(num_epochs):
            loss_val = 0
            print(len(index_pairing))
            for data, target in index_pairing:
                x = self.get_input_layer(data)
                y_true = torch.tensor([target], device=self._device, dtype=torch.long)

                log_softmax_temp = log_softmax(self.f(x), dim=0)

                loss = nll_loss(log_softmax_temp.view(1, -1), y_true)
                loss_val += loss.data.item()
                loss.backward()
                self._W1.data -= learning_rate * self._W1.grad.data
                self._W2.data -= learning_rate * self._W2.grad.data

                self._W1.grad.data.zero_()
                self._W2.grad.data.zero_()
            if verbose:
                print(f'Loss at epo {epo}: {loss_val / len(index_pairing)}')
