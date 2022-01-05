import numpy as np
import pytorch_network
import torch
from torch import nn
import tqdm

from utils import get_aligned_representations
from information_process import get_information
from plot_information import plot_information_plane
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# %matplotlib inline
from pdb import set_trace as st 

model = pytorch_network.MLPWithInfo(output_activation=None)

print(model)

print(model.info_layers_numbers)

X, y = pytorch_network.load_tishby_toy_dataset('./data/g1.mat')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)


epochs = 1000
train_res = pytorch_network.train_network(model, X_train, y_train.astype(np.int),
                                          X_test, y_test.astype(np.int), batch_size=12, epochs=epochs)

ws = model.representations_per_epochs
order = train_res[2]

ws = get_aligned_representations(ws, order)

assert len(model.representations_per_epochs) == epochs
assert len(model.representations_per_epochs[0]) == len(model.info_layers_numbers)

for i in range(len(model.representations_per_epochs[0])):
    assert(model.representations_per_epochs[0][i].shape[0] == X_train.shape[0])

plt.plot(np.arange(len(train_res[0])), train_res[0])

plt.plot(np.arange(len(train_res[1])), train_res[1])

num_of_bins = 40
every_n = 10
IXT_array, ITY_array = get_information(ws, X_train, np.concatenate([y_train, 1 - y_train], axis=1), 
                                       num_of_bins, every_n=every_n, return_matrices=True)

import importlib
import plot_information
importlib.reload(plot_information)
plot_information.plot_information_plane(IXT_array, ITY_array, num_epochs=epochs, every_n=every_n)