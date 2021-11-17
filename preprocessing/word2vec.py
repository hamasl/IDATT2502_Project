# Implementation inspired by: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
import torch
from torch.nn.functional import log_softmax, nll_loss
from tqdm import tqdm


class Word2Vec:

    def __init__(self, vocab_size, word_idx, device=torch.device("cpu")):
        self.vocab_size = vocab_size
        self.word_idx = word_idx
        self.embedding_dims = 5
        self._device = device
        self._W1 = torch.randn(self.embedding_dims, self.vocab_size, device=device, requires_grad=True,
                               dtype=torch.float)
        self._W2 = torch.randn(self.vocab_size, self.embedding_dims, device=device, requires_grad=True,
                               dtype=torch.float)

    def get_input_layer(self, word_idx):
        """
        Creates the input layer.
        :param word_idx:
        :return:
        """
        x = torch.zeros(self.vocab_size, device=self._device, dtype=torch.float)
        x[word_idx] = 1.0
        return x

    def f(self, x):
        return torch.matmul(self._W2, torch.matmul(self._W1, x))

    def similarity(self, word_idx1, word_idx2):
        """
        Measures the similarity of two words.
        :param word_idx1: The index of word1.
        :param word_idx2: The index of word2.
        :return: Int between 1 and -1, where 1 i equal.
        """
        return torch.dot(self._W2[word_idx1], self._W2[word_idx2]) / (
                    torch.norm(self._W2[word_idx1]) * torch.norm(self._W2[word_idx2]))

    def train(self, num_epochs, learning_rate, index_pairing, verbose=False):
        """
        Trains the word2vec model.
        :param num_epochs: The number of epochs to run.
        :param learning_rate: The learning rate while training.
        :param index_pairing: The data used to train the word2vec model.
        :param verbose: If True loss is printed after each epoch.
        :return:
        """
        for epo in range(num_epochs):
            loss_val = 0
            for index, (data, target) in enumerate(tqdm(index_pairing)):
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
