import argparse
import sys
from utils import *

parser = argparse.ArgumentParser(description = 'information_plane')

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--ipt_dim', type=int, default=12)
parser.add_argument('--layers_dim', type=str, default='[10, 7, 5, 4, 3, 1]')
parser.add_argument('--hidden_act', type=str, default='tanh')
parser.add_argument('--opt_act', type=str, default='iden')
parser.add_argument('--last_act', type=str, default='sigmoid')
parser.add_argument('--plot_interval', type=int, default=10)
parser.add_argument('--marksize', type=int, default=15)
# revise
args = parser.parse_args()






