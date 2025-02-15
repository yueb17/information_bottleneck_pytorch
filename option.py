import argparse
import sys
from utils import *
from smilelogging import Logger
from smilelogging import argparser as parser

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--ipt_dim', type=int, default=12)
parser.add_argument('--layers_dim', type=str, default='[10, 7, 5, 4, 3, 1]')
parser.add_argument('--hidden_act', type=str, default='tanh')
parser.add_argument('--opt_act', type=str, default='iden')
parser.add_argument('--last_act', type=str, default='sigmoid')
parser.add_argument('--plot_interval', type=int, default=10)
parser.add_argument('--marksize', type=int, default=15)
parser.add_argument('--bin_num', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.97)
parser.add_argument('--loss', type=str, default='bce')
parser.add_argument('--bin_min', type=int, default=-1)
parser.add_argument('--bin_max', type=int, default=1)
parser.add_argument('--PI', type=str, default='dense', choices=['dense', 'random', 'magnitude'])
# revise
args = parser.parse_args()
logger = Logger(args)
gen_img_path = logger.gen_img_path
weights_path = logger.weights_path
global print; print = logger.log_printer.logprint






