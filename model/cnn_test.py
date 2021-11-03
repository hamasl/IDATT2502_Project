import torch
from torch import nn

if __name__ == '__main__':

    num_classes = 10
    a = torch.rand(6000, 1, 32)
    size = a.shape[2]
    # cnn_multiple is connected to MaxPool layer. Which is kernel_size**num_of_max_pool_layers
    cnn_multiple = 4
    if size % cnn_multiple != 0:
        raise Exception(f"Size of each function needs to be a multiple of {cnn_multiple}")
    print(size//4)
    pipe = [nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * size//4, 1024),
            nn.Linear(1024, num_classes)]

    print(a.size())

    for i in range(len(pipe)):
        a = pipe[i](a)
        print(pipe[i])
        print(a.size())
