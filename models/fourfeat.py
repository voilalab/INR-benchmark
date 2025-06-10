import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from model_utils import calc_layer_width

torch.set_default_dtype(torch.float32)

class InputMapping(nn.Module):
    def __init__(self, B, dimension):
        super().__init__()
        
        # self.B = B
        
        self.register_buffer('B', B)
        
        self.dimension = dimension

    # Fourier feature mapping (https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb#scrollTo=OcJUfBV0dCww)
    def forward(self, x):
        x_proj = (2.*np.pi*x) @ self.B.T # B,2 x 2,1000
        result = torch.concatenate([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        
        return result

# Fourier Features
class FourFeat(nn.Module):
    def __init__(self, dimension, max_params, out_features = 1, hidden_out = False, mapping_sigma = 20, num_hidden_layers = 2, mapping_size=1000):
        super().__init__()
        # Choose layer width for FourFeat so that total model size obeys an upper bound
        in_features = dimension
        self.out_features = out_features

        # in fourier features embedding process makes the embedded shape always 2*mapping size independent to dimension
        layer_width = calc_layer_width(2, self.out_features, num_hidden_layers, mapping_size, max_params, is_dict=hidden_out)

        # Create the random features for this model
        self.dimension = dimension
        self.mapping_size = mapping_size
        self.mapping_sigma = mapping_sigma
        self.register_buffer('B', torch.tensor(
                np.random.normal(size=(self.mapping_size, self.dimension)),
                dtype=torch.float
            ) * self.mapping_sigma)
        
        # Create the MLP with the input embedding
        self.model = nn.Sequential()
        self.model.add_module("fourfeat", InputMapping(self.B, dimension=self.dimension))
        self.model.add_module("dense", nn.Linear(self.mapping_size*2, layer_width))
        self.num_params = self.mapping_size * 2 * layer_width
        for i in range(num_hidden_layers):
            self.model.add_module(f"act{i}", nn.ReLU())
            self.model.add_module(f"dense{i}", nn.Linear(layer_width, layer_width))
            self.num_params = self.num_params + layer_width * layer_width
        if hidden_out:
            self.out_features = layer_width
        self.model.add_module("output", nn.Linear(layer_width, self.out_features))
        self.num_params = self.num_params + layer_width
        tqdm.write(f'layer_width: {layer_width}, num params: {self.num_params}, sigma: {mapping_sigma}')

    def forward(self, x):
        x = self.model(x)
        return x

import time
if __name__ == '__main__':
    dimension = 3
    for model_size in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]:
        inr = FourFeat(dimension, model_size, 10)
        print(inr)
        coords = np.linspace(0, 1, 100, endpoint=False)
        x = torch.tensor(np.stack(np.meshgrid(*[coords for _ in range(dimension)]), -1))