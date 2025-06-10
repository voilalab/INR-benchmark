import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# 2, 1, , 1, max_params
def calc_layer_width(in_features, out_features, num_hidden_layers, mapping_size, max_params,  num_models = 4, is_dict = False):
    if not is_dict:
        a = num_hidden_layers
        if mapping_size >= 100:
            b = 2*mapping_size + out_features + num_hidden_layers
            in_features = 2*mapping_size
        else:
            b = in_features + out_features + num_hidden_layers
    else:
        tqdm.write("Calulating for the New dictionary model")
        a = (num_hidden_layers-1)*num_models**2
        b = 2*mapping_size + in_features*(num_models-1) + (num_models -1)
    c = in_features + out_features -max_params
    width = (-b + np.sqrt(b*b - 4*a*c)) / (2*a) # formula of roots
    layer_width = int(np.floor(width))
    return layer_width
