import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from pdb import set_trace as st
import os
from smilelogging import Logger
from option import *

matplotlib.rcParams.update({'font.size': 18})


def plot_information_plane(IXT_array, ITY_array, num_epochs, every_n, args):
    assert len(IXT_array) == len(ITY_array)

    max_index = len(IXT_array)

    plt.figure(figsize=(15, 9))
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_epochs + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[i, :]
        ITY = ITY_array[i, :]
        plt.plot(IXT, ITY, marker='o', markersize=args.marksize, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i * every_n], zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num epochs')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(num_epochs), transform=cbar.ax.transAxes, va='bottom', ha='center')
    
    IP_path = gen_img_path + '/epoch_wise.jpg'
    plt.savefig(IP_path)

    plt.show()

def plot_information_plane2(IXT_array, ITY_array, num_epochs, every_n, args):
    assert len(IXT_array) == len(ITY_array)

    max_index = np.shape(IXT_array)[1]

    plt.figure(figsize=(15, 9))
    plt.xlabel('$I(X;T)$')
    plt.ylabel('$I(T;Y)$')

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, max_index + 1)]

    for i in range(0, max_index):
        IXT = IXT_array[:, i]
        ITY = ITY_array[:, i]
        plt.plot(IXT, ITY, marker='o', markersize=args.marksize, markeredgewidth=0.04,
                 linestyle=None, linewidth=1, color=colors[i], zorder=10)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar = plt.colorbar(sm, ticks=[])
    cbar.set_label('Num layers')
    cbar.ax.text(0.5, -0.01, 0, transform=cbar.ax.transAxes, va='top', ha='center')
    cbar.ax.text(0.5, 1.0, str(max_index), transform=cbar.ax.transAxes, va='bottom', ha='center')
    
    IP_path = gen_img_path + '/layer_wise.jpg'
    plt.savefig(IP_path)

    plt.show()