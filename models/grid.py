import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from bandlimited_signal import RealImage
torch.set_default_dtype(torch.float)
# torch.manual_seed(1234)

# From https://github.com/sarafridov/K-Planes/blob/main/plenoxels/ops/interpolation.py
def grid_sample_wrapper(
    grid: torch.Tensor,
    coords: torch.Tensor,
    interpolation='bilinear',
    align_corners: bool = True,
    out_features=1,
) -> torch.Tensor:
    grid_dim = coords.shape[-1]
    grid = grid.unsqueeze(0)

    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
        if grid_dim == 3:
            coords = coords.unsqueeze(0)
    else:
        raise NotImplementedError(
            f"Grid-sample was called with {grid_dim}D data but is only "
            f"implemented for 2 and 3D data."
        )

    coords = coords.unsqueeze(0)

    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    feature_dim = out_features
    interp = grid_sampler(
        grid,
        coords,
        align_corners=align_corners,
        mode=interpolation,
        padding_mode='border',
    )

    interp = interp.reshape(B, feature_dim, n).transpose(-1, -2)
    interp = interp.squeeze()
    return interp


# Grid representations
class Grid(nn.Module):
    def __init__(self, dimension, max_params, out_features=1, interpolation='bicubic'):
        super().__init__()

        # max_params here is the per-output-channel grid budget.
        # For RGB, pass total_budget / 3 from the caller.
        resolution = int(np.floor(max_params ** (1.0 / dimension)))
        tqdm.write(f'resolution is {resolution}')

        self.out_features = out_features
        self.dimension = dimension

        self.grid = nn.Parameter(
            torch.ones(*([out_features] + [resolution] * dimension)) * 0.1
        )

        # Actual trainable parameter count includes output channels.
        self.num_params = out_features * (resolution ** dimension)

        if dimension == 3:
            self.interpolation = 'bilinear'
        else:
            self.interpolation = interpolation

    def forward(self, x):
        result = grid_sample_wrapper(
            grid=self.grid,
            coords=x.reshape(-1, x.shape[-1]),
            interpolation=self.interpolation,
            out_features=self.out_features,
        )
        result = result.reshape((-1, self.out_features)).squeeze()
        return result


def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    dimension = 3
    out_features = 3  # RGB. Use 1 for grayscale/occupancy.

    for model_size in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]:
        constructor_max_params = model_size / out_features

        inr = Grid(
            dimension,
            max_params=constructor_max_params,
            out_features=out_features,
        )

        print(
            f'target_budget={model_size:.0f}, '
            f'constructor_max_params={constructor_max_params:.0f}, '
            f'actual_params={count_parameters(inr)}'
        )
