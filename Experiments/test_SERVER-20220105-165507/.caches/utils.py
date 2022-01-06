import numpy as np
from torch import nn

def get_aligned_representations(representations, order):
    for epoch in range(len(representations)):
        for layer in range(len(representations[0])):
            representations[epoch][layer] = representations[epoch][layer][np.argsort(order[epoch]), :]

    return representations

def strlist_to_list(sstr, ttype):
    '''
        example:
        # self.args.stage_pr = [0, 0.3, 0.3, 0.3, 0, ]
        # self.args.skip_layers = ['1.0', '2.0', '2.3', '3.0', '3.5', ]
        turn these into a list of <ttype> (float or str or int etc.)
    '''
    if not sstr:
        return sstr
    out = []
    sstr = sstr.strip()
    if sstr.startswith('[') and sstr.endswith(']'):
        sstr = sstr[1:-1]
    for x in sstr.split(','):
        x = x.strip()
        if x:
            x = ttype(x)
            out.append(x)
    return out

def act_dict(act_name):
    dict_ = {'relu': nn.ReLU,
                'softmax': nn.Softmax,
                'tanh': nn.Tanh,
                'sigmoid': nn.Sigmoid,
                'iden': None,
                }

    return dict_[act_name]