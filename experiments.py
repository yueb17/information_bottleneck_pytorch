import numpy as np
import pytorch_network
import torch
from torch import nn
import tqdm

from utils import get_aligned_representations
from information_process import get_information
from plot_information import plot_information_plane, plot_information_plane2
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# %matplotlib inline
from pdb import set_trace as st 
from option import args
from option import *
from utils import *
from smilelogging import Logger

print(args)

model = pytorch_network.MLPWithInfo( 
	input_dim=args.ipt_dim,
	layers_dim=strlist_to_list(args.layers_dim, int),
	activation=act_dict(args.hidden_act),
	output_activation=act_dict(args.opt_act), 
	last_activation=act_dict(args.last_act)
	)

print(model)

print(model.info_layers_numbers)

X, y = pytorch_network.load_tishby_toy_dataset('./data/g1.mat')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, train_size=0.9)


train_res = pytorch_network.train_network(model, X_train, y_train.astype(np.int),
                                          X_test, y_test.astype(np.int), args=args, batch_size=args.batch_size, epochs=args.epoch)

ws = model.representations_per_epochs
order = train_res[3]
# st()
ws = get_aligned_representations(ws, order)

assert len(model.representations_per_epochs) == args.epoch
assert len(model.representations_per_epochs[0]) == len(model.info_layers_numbers)

for i in range(len(model.representations_per_epochs[0])):
    assert(model.representations_per_epochs[0][i].shape[0] == X_train.shape[0])

fig, axs = plt.subplots(2, 2)
axs[0,0].plot(np.arange(len(train_res[0])), train_res[0]), axs[0,0].set_title('train_loss')
axs[0,1].plot(np.arange(len(train_res[1])), train_res[1]), axs[0,1].set_title('test_acc')
axs[1,0].plot(np.arange(len(train_res[2])), train_res[2]), axs[1,0].set_title('train_acc')

acc_path = gen_img_path + '/acc.jpg'
plt.savefig(acc_path)


num_of_bins = args.bin_num
every_n = args.plot_interval
# st()
IXT_array, ITY_array = get_information(ws, X_train, np.concatenate([y_train, 1 - y_train], axis=1), 
                                       num_of_bins, args, every_n=every_n, return_matrices=True)

import importlib
import plot_information
importlib.reload(plot_information)
plot_information.plot_information_plane(IXT_array, ITY_array, num_epochs=args.epoch, every_n=every_n, args=args)
plot_information.plot_information_plane2(IXT_array, ITY_array, num_epochs=args.epoch, every_n=every_n, args=args)

