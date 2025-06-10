import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import numpy as np
from functools import partial
from torch import nn
import copy


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1/math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class Sine(nn.Module):
    def __init__(self, w0=20):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input)-self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


class MFNBase(nn.Module):

    def __init__(self, hidden_size, out_size, n_layers, weight_scale,
                 bias=True, output_act=False):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )

        self.output_linear = nn.Linear(hidden_size, out_size)

        self.output_act = output_act

        self.linear.apply(mfn_weights_init)
        self.output_linear.apply(mfn_weights_init)

    def forward(self, model_input):

        input_dict = {key: input.clone().detach().requires_grad_(True)
                      for key, input in model_input.items()}
        coords = input_dict['coords']

        out = self.filters[0](coords)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return {'model_in': input_dict, 'model_out': {'output': out}}


class FourierLayer(nn.Module):

    def __init__(self, in_features, out_features, weight_scale, quantization_interval=2*np.pi):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        r = 2*weight_scale[0] / quantization_interval
        assert math.isclose(r, round(r)), \
               'weight_scale should be divisible by quantization interval'

        # sample discrete uniform distribution of frequencies
        for i in range(self.linear.weight.data.shape[1]):
            init = torch.randint_like(self.linear.weight.data[:, i],
                                      0, int(2*weight_scale[i] / quantization_interval)+1)
            init = init * quantization_interval - weight_scale[i]
            self.linear.weight.data[:, i] = init

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class BACON(MFNBase):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size,
                 hidden_layers=3,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,  # assumes data range [-.5,.5]
                 centered=True,
                 input_scales=None,
                 output_layers=None,
                 is_sdf=False,
                 reuse_filters=False,
                 **kwargs):

        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.centered = centered
        self.frequency = frequency
        self.is_sdf = is_sdf
        self.reuse_filters = reuse_filters
        self.in_size = in_size

        # we need to multiply by this to be able to fit the signal
        input_scale = [round((np.pi * freq / (hidden_layers + 1))
                       / quantization_interval) * quantization_interval for freq in frequency]

        self.filters = nn.ModuleList([
                FourierLayer(in_size, hidden_size, input_scale,
                             quantization_interval=quantization_interval)
                for i in range(hidden_layers + 1)])

        print(self)

    def forward_mfn(self, input_dict):
        if 'coords' in input_dict:
            coords = input_dict['coords']
        elif 'ray_samples' in input_dict:
            if self.in_size > 3:
                coords = torch.cat((input_dict['ray_samples'], input_dict['ray_orientations']), dim=-1)
            else:
                coords = input_dict['ray_samples']

        if self.reuse_filters:
            filter_outputs = 3 * [self.filters[2](coords), ] + \
                             2 * [self.filters[4](coords), ] + \
                             2 * [self.filters[6](coords), ] + \
                             2 * [self.filters[8](coords), ]

            out = filter_outputs[0]
            for i in range(1, len(self.filters)):
                out = filter_outputs[i] * self.linear[i - 1](out)
        else:
            out = self.filters[0](coords)
            for i in range(1, len(self.filters)):
                out = self.filters[i](coords) * self.linear[i - 1](out)

        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out

    def forward(self, model_input, mode=None, integral_dim=None):

        out = {'output': self.forward_mfn(model_input)}

        if self.is_sdf:
            return {'model_in': model_input['coords'],
                    'model_out': out['output']}

        return {'model_in': model_input, 'model_out': out}


class MultiscaleBACON(MFNBase):
    def __init__(self,
                 in_size,
                 hidden_size,
                 out_size = 1,
                 hidden_layers=3,
                 weight_scale=1.0,
                 bias=True,
                 output_act=False,
                 frequency=(128, 128),
                 quantization_interval=2*np.pi,
                 centered=True,
                 is_sdf=False,
                 input_scales=None,
                 output_layers=None,
                 reuse_filters=False):

        super().__init__(hidden_size, out_size, hidden_layers,
                         weight_scale, bias, output_act)

        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.centered = centered
        self.is_sdf = is_sdf
        self.frequency = frequency
        self.output_layers = output_layers
        self.reuse_filters = reuse_filters
        self.stop_after = None

        # we need to multiply by this to be able to fit the signal
        if input_scales is None:
            input_scale = [round((np.pi * freq / (hidden_layers + 1))
                           / quantization_interval) * quantization_interval for freq in frequency]

            self.filters = nn.ModuleList([
                    FourierLayer(in_size, hidden_size, input_scale,
                                 quantization_interval=quantization_interval)
                    for i in range(hidden_layers + 1)])
        else:
            if len(input_scales) != hidden_layers+1:
                raise ValueError('require n+1 scales for n hidden_layers')
            input_scale = [[round((np.pi * freq * scale) / quantization_interval) * quantization_interval
                           for freq in frequency] for scale in input_scales]

            self.filters = nn.ModuleList([
                           FourierLayer(in_size, hidden_size, input_scale[i],
                                        quantization_interval=quantization_interval)
                           for i in range(hidden_layers + 1)])

        # linear layers to extract intermediate outputs
        self.output_linear = nn.ModuleList([nn.Linear(hidden_size, out_size) for i in range(len(self.filters))])
        self.output_linear.apply(mfn_weights_init)

        # if outputs layers is None, output at every possible layer
        if self.output_layers is None:
            self.output_layers = np.arange(1, len(self.filters))

        # print(self)

    def layer_forward(self, coords, filter_outputs, specified_layers,
                      get_feature, continue_layer, continue_feature):
        """ for multiscale SDF extraction """

        # hardcode the 8 layer network that we use for all sdf experiments
        filter_ind_dict = [2, 2, 2, 4, 4, 6, 6, 8, 8]
        outputs = []

        if continue_feature is None:
            assert(continue_layer == 0)
            out = self.filters[filter_ind_dict[0]](coords)
            filter_output_dict = {filter_ind_dict[0]: out}
        else:
            out = continue_feature
            filter_output_dict = {}

        for i in range(continue_layer+1, len(self.filters)):
            if filter_ind_dict[i] not in filter_output_dict.keys():
                filter_output_dict[filter_ind_dict[i]] = self.filters[filter_ind_dict[i]](coords)
            out = filter_output_dict[filter_ind_dict[i]] * self.linear[i - 1](out)

            if i in self.output_layers and i == specified_layers:
                if get_feature:
                    outputs.append([self.output_linear[i](out), out])
                else:
                    outputs.append(self.output_linear[i](out))
                return outputs

        return outputs

    def forward(self, model_input, specified_layers=None, get_feature=False,
                continue_layer=0, continue_feature=None):

        if self.is_sdf:
            model_input = {key: input.clone().detach().requires_grad_(True)
                           for key, input in model_input.items()}

        # if 'coords' in model_input:
        coords = model_input
        # elif 'ray_samples' in model_input:
        #     coords = model_input['ray_samples']

        outputs = []
        if self.reuse_filters:

            # which layers to reuse
            if len(self.filters) < 9:
                filter_outputs = 2 * [self.filters[0](coords), ] + \
                        (len(self.filters)-2) * [self.filters[-1](coords), ]
            else:
                filter_outputs = 3 * [self.filters[2](coords), ] + \
                                 2 * [self.filters[4](coords), ] + \
                                 2 * [self.filters[6](coords), ] + \
                                 2 * [self.filters[8](coords), ]

            # multiscale sdf extractions (evaluate only some layers)
            if specified_layers is not None:
                outputs = self.layer_forward(coords, filter_outputs, specified_layers,
                                             get_feature, continue_layer, continue_feature)

            # evaluate all layers
            else:
                out = filter_outputs[0]
                for i in range(1, len(self.filters)):
                    out = filter_outputs[i] * self.linear[i - 1](out)

                    if i in self.output_layers:
                        outputs.append(self.output_linear[i](out))
                        if self.stop_after is not None and len(outputs) > self.stop_after:
                            break

        # no layer reuse
        else:
            out = self.filters[0](coords)
            for i in range(1, len(self.filters)):
                out = self.filters[i](coords) * self.linear[i - 1](out)

                if i in self.output_layers:
                    outputs.append(self.output_linear[i](out))
                    if self.stop_after is not None and len(outputs) > self.stop_after:
                        break

        if self.is_sdf:  # convert dtype
            return {'model_in': model_input['coords'],
                    'model_out': outputs}  # outputs is a list of tensors
        
        # print(model_input.max(),model_input.min(), model_input.shape, len(outputs))
        # print(outputs[0].shape, outputs[1].shape, outputs[2].shape)
        # return {'model_in': model_input, 'model_out': {'output': outputs}}
        return outputs
    
class BACON(nn.Module):
    def __init__(self, dimension , max_params, out_features = 1, num_hidden_layers = 4 ,resolution = [1000,1000]):
        super().__init__()
        out_features = out_features
        self.resolution = resolution
        
        m = MultiscaleBACON
        if dimension == 2:
            self.hidden_layers = num_hidden_layers = 4
            layer_width_dict = {1e4:48, 3e4:85, 1e5:156, 3e5:272, 1e6:498, 3e6:864}

            self.hidden_features = layer_width_dict[max_params]

            input_scales = [1/8, 1/8, 1/4, 1/4, 1/4]
            output_layers = [1, 2, 4]
            self.model = model = m(dimension, self.hidden_features, out_size=out_features,
                    hidden_layers=self.hidden_layers,
                    bias=True,
                    frequency=(self.resolution[0], self.resolution[1]),
                    quantization_interval=2*np.pi,
                    input_scales=input_scales,
                    output_layers=output_layers,
                    reuse_filters=False)
        elif dimension == 3:
            self.hidden_layers = num_hidden_layers = 8
            layer_width_dict = {1e4:34, 3e4:60, 1e5:110, 3e5:192, 1e6:352, 3e6:611}

            self.hidden_features = layer_width_dict[max_params]
            # self.hidden_features = max_params
            input_scales = [1/24, 1/24, 1/24, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
            output_layers = [2, 4, 6, 8]
            self.model = model = m(dimension, self.hidden_features, out_size=out_features,
                    hidden_layers=self.hidden_layers,
                    bias=True,
                    frequency=self.resolution,
                    quantization_interval=2*np.pi,
                    input_scales=input_scales,
                    output_layers=output_layers,
                    reuse_filters=True)
        

        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())

        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print(f'Num. Parameters: {params} for bacon')

    def forward(self, x):
        output = self.model(x)
        return output
    
def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)

from tqdm import tqdm
if __name__ == '__main__':
    dimension = 3
    target_params = [int(1e4), int(3e4), int(1e5), int(3e5), int(1e6), int(3e6)]
    closest_configs = []

    for target in target_params:
        best_diff = float('inf')
        best_hidden = None
        best_actual = None

        # hidden_feature candidate
        for hidden in tqdm(range(8, 2048)):
            model = BACON(dimension, hidden, resolution = [100,100,100])  # input_dim = 2
            params = count_parameters(model)
            diff = abs(params - target)
            if diff < best_diff:
                best_diff = diff
                best_hidden = hidden
                best_actual = params
            if diff == 0:
                break  # if it matches exactly

        closest_configs.append((target, best_hidden, best_actual))

    # output
    for target, h, actual in closest_configs:
        print(f"Target: {target}, Hidden: {h}, Actual Params: {actual}")
    # dimension = 3
    # model = BACON(dimension, 1e4, resolution=[100,100,100])
    # coords = np.linspace(0, 1, 100, endpoint=False)
    # x = torch.tensor(np.stack(np.meshgrid(*[coords for _ in range(dimension)]), -1), dtype = torch.float32)
    # output = model(x)
    # print(output[-1].shape)