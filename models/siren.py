import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from model_utils import *

torch.set_default_dtype(torch.float)

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

# SIREN
class Siren(nn.Module):
    def __init__(self, dimension, max_params, out_features = 1, hidden_out = False, mapping_sigma = 90, num_hidden_layers=3, outermost_linear=False):
        super().__init__()
        self.dimension = in_features = dimension
        self.out_features = out_features
        mapping_size = 1
        layer_width = calc_layer_width(in_features, self.out_features, num_hidden_layers, mapping_size, max_params, is_dict=hidden_out)
        hidden_features = layer_width

        message = f'layer_width: {layer_width}'
        tqdm.write(message)

        '''Gotta fix here someday'''
        if mapping_sigma == None or mapping_sigma == 0:
            first_omega_0 = hidden_omega_0 = omega_0
        else: first_omega_0 = hidden_omega_0 = omega_0 = mapping_sigma

        self.model = []
        self.model.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(num_hidden_layers):
            self.model.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if hidden_out:
            self.out_features = hidden_features
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, self.out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.model.append(final_linear)
        else:
            self.model.append(SineLayer(hidden_features, self.out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.model = nn.Sequential(*self.model)

    def forward(self,x):
        x = self.model(x)
        return x
    
def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    dimension = 3
    max_params = 1e6
    sigma = 1
    for model_size in [1e4,3e4,1e5,3e5,1e6,3e6]:
        inr = Siren(dimension=dimension, max_params=model_size, mapping_sigma=sigma, num_hidden_layers=3)
        
        print(inr, count_parameters(inr))
    