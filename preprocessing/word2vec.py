# Implementation inspired by: https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
import os

import torch
from torch.nn.functional import log_softmax, nll_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class Word2Vec:

    def __init__(self, vocab_size, word2idx, idx2word, device=torch.device("cpu")):
        self.vocab_size = vocab_size
        self.word2idx = word2idx
        self.idx2word = idx2word
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
        :return: Int between 1 and -1, where 1 means equal words. Cosine similarity
        """
        w1v = torch.matmul(self._W2, torch.matmul(self._W1, self.get_input_layer(word_idx1)))
        w2v = torch.matmul(self._W2, torch.matmul(self._W1, self.get_input_layer(word_idx2)))
        return torch.dot(w1v, w2v) / (torch.norm(w1v) * torch.norm(w2v))

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

    def get_closest_word(self, word, top_n=5):
        """
        Finds top_n closest words from a given word
        based on the Word2Vec model
        :param word: chosen word
        :param top_n: amount of closest words to be returned
        :return: array of tuples, with nearest words and their cosine similarity value
        """
        word_distance = []
        i = self.word2idx[word]
        for j in range(self.vocab_size):
            if j != i:
                word_distance.append((self.idx2word[j], self.similarity(j, i)))
        word_distance.sort(key=lambda x: x[1], reverse=True)
        return word_distance[:top_n]

    def plot2D(self, similarity_table: [[]]):
        """
        Plots word vectors in a 2D space.
        Also saves the plot as .png file in preprocessing/plots folder
        :param similarity_table: 2D array
        :return:
        """
        trans_table = [list(i) for i in zip(*similarity_table)]
        pca = PCA(n_components=2)
        pca.fit(trans_table)
        Xpca = pca.transform(trans_table)
        for i in range(len(self.idx2word)):
            plt.scatter(Xpca[i, 0], Xpca[i, 1])
            plt.annotate(self.idx2word[i], (Xpca[i, 0], Xpca[i, 1]), alpha=0.7)
        # Disable axis visibility for better readability
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        # Create plots folder (if it does not exists)
        dirname = os.path.join(os.path.dirname(__file__), "../preprocessing/plots")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(os.path.join(dirname, "2D-plot.png"))
        plt.show()

    def plot3D(self, similarity_table: [[]]):
        """
        Plots word vectors in a 3D space
        Also saves the plot as .png file in preprocessing/plots folder
        :param similarity_table: 2D array
        :return:
        """
        trans_table = [list(i) for i in zip(*similarity_table)]
        pca = PCA(n_components=3)
        pca.fit(trans_table)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        Xpca = pca.transform(trans_table)
        for i in range(len(self.idx2word)):
            ax.scatter(Xpca[i, 0], Xpca[i, 1], Xpca[i, 2])
            ax.text(Xpca[i, 0], Xpca[i, 1], Xpca[i, 2], self.idx2word[i], alpha=0.7)
        # Disable axis visibility for better readability
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Create plots folder (if it does not exists)
        dirname = os.path.join(os.path.dirname(__file__), "../preprocessing/plots")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(os.path.join(dirname, "3D-plot.png"))
        plt.show()
