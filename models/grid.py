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
def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, interpolation = 'bilinear', align_corners: bool = True, out_features = 1) -> torch.Tensor:
    grid_dim = coords.shape[-1]
    grid = grid.unsqueeze(0)
    # if grid.dim() == grid_dim + 1:
    #     # no batch dimension present, need to add it
    #     grid = grid.unsqueeze(1)
        
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
        # coords = torch.cat([coords]*out_features, axis = 0)
        # coords = coords.reshape( 1, -1, coords.shape[-1])
        # print(coords.shape)

    if grid_dim == 2 or grid_dim == 3:
        grid_sampler = F.grid_sample
        if grid_dim == 3:
            coords = coords.unsqueeze(0)
    else:
        raise NotImplementedError(f"Grid-sample was called with {grid_dim}D data but is only "
                                  f"implemented for 2 and 3D data.")
    # coords = coords.reshape([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    # coords = coords.view(1, coords.shape[1], coords.shape[2], coords.shape[-1])
    coords = coords.unsqueeze(0)

    # print(f"Grid shape: {grid.shape}, Coords shape: {coords.shape}")
    B, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    feature_dim = out_features # I don't know why did it worked but it works.
    # print('coords', coords.shape)
    interp = grid_sampler(
        grid,  # [B, feature_dim, reso, ...]  [1,1,1,1000]
        coords,  # [B, 1, ..., n, grid_dim]  [1,1,10000,2]
        align_corners=align_corners,
        mode=interpolation, padding_mode='border')
    # interp -> [1,1,1,10000]
    # print(interp.shape)
    interp = interp.reshape(B, feature_dim, n).transpose(-1, -2)  # [B, n, feature_dim]
    interp = interp.squeeze()  # [B?, n, feature_dim?]
    return interp

# Grid representations
class Grid(nn.Module):
    def __init__(self, dimension, max_params, out_features = 1, interpolation = 'bicubic'):
        super().__init__()
        # Choose the resolution to obey max_params
        resolution = int(np.floor(max_params**(1.0/dimension)))
        tqdm.write(f'resolution is {resolution}')
        self.out_features = out_features
        # Create the grid
        self.dimension = dimension
        self.grid = nn.Parameter(torch.ones(*([out_features,] + [resolution] * dimension)) * 0.1)
        # print(self.grid.shape)
        # self.grid = nn.Parameter(torch.ones(1, resolution, resolution) * 0.1)

        self.num_params = resolution ** dimension
        if dimension == 3:
            self.interpolation = 'bilinear'    
        else:
            self.interpolation = interpolation
        # Set up for optimization
        # self.loss_fn = nn.MSELoss()
        # self.optimizer = torch.optim.Adam([self.grid], lr=0.1)


    def forward(self, x):
        # Use interpolation to retrieve the value of x within the grid
        # print('forward inside grid',x.shape)
        # result = grid_sample_wrapper(grid=self.grid, coords=x.reshape(-1, x.shape[-1]), interpolation = self.interpolation).reshape(x.shape[:-1])
        # print(self.grid.shape, x.reshape(-1, x.shape[-1]))
        result = grid_sample_wrapper(grid=self.grid, coords=x.reshape(-1, x.shape[-1]), interpolation = self.interpolation, out_features = self.out_features)
        result = result.reshape((-1 ,self.out_features)).squeeze()
        # print('result',result.shape)
        return result
    

def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    dimension = 3
    max_params = 1e6
    sigma = 1
    for model_size in [1e4,3e4,1e5,3e5,1e6,3e6]:
        inr = Grid(dimension, max_params=model_size)
        
        print(inr, count_parameters(inr))

# if __name__ == '__main__':
#     max_param = 1e4
#     dimension = 2
#     loss_fn = nn.MSELoss()
#     signal = RealImage(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution = False)
#     signal = torch.tensor(signal.signal, dtype=torch.float).cuda()
#     coords_y = np.linspace(-1, 1, signal.shape[0], endpoint=False)
#     coords_x = np.linspace(-1, 1, signal.shape[1], endpoint=False)
#     x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y), -1), dtype=torch.float).cuda()

#     num_samples = signal.shape[0]*signal.shape[1]
#     signal_shape = signal.shape
#     print(signal_shape)
#     x = x.reshape(-1, dimension)
#     signal = signal.reshape(num_samples,3).squeeze()
#     # output_ = []
#     # for i in range(3):
#     #     grid = Grid(dimension=dimension,max_params=max_param//3, out_features = 1,interpolation='bilinear').cuda()
#     #     optim_grid = torch.optim.Adam(lr=1e-1, params=[grid.grid])
#     #     num_steps = 1000
#     #     for iter in tqdm(range(num_steps), desc='grid', leave=False):
#     #         output = grid(x)
#     #         loss = loss_fn(output, signal[:,i])
#     #         optim_grid.zero_grad()
#     #         loss.backward()
#     #         optim_grid.step()  
#     #         if iter == num_steps -1:
#     #             with torch.no_grad():
#     #                 output = grid(x)
#     #                 loss = loss_fn(output, signal[:,i])
#     #                 loss_val = loss_fn(output, signal[:,i]).cpu().squeeze()
#     #                 psnr = -10*torch.log10(loss_val).numpy()
#     #                 grid_psnr = psnr
#     #     print(output.shape)
        
#     #     plt.figure()
#     #     output__ = np.clip(output.squeeze().detach().cpu().numpy(), 0, 1)
#     #     output__ = output__.reshape(339,510)
#     #     output_.append(output__)
#     #     # output2save = (output2save - output2save.min()) / (output2save.max() - output2save.min())
#     #     plt.imshow(output__)
#     #     plt.title(f'PSNR: {psnr}')
#     #     plt.tight_layout()
#     #     # plt.savefig(f'{model_name}_{bandlimit}.png')
#     #     plt.savefig(f'{max_param:0.0e}_{i}.png')
#     #     plt.close()
#     # # output_ = np.array(output_)
#     # output2save = output_
#     # torch.cuda.empty_cache()
    
#     grid = Grid(dimension=dimension,max_params=max_param, out_features = 3,interpolation='bilinear').cuda()
#     optim_grid = torch.optim.Adam(lr=1e-1, params=[grid.grid])
#     num_steps = 1000
#     for iter in tqdm(range(num_steps), desc='grid', leave=False):
#         output = grid(x)
#         loss = loss_fn(output, signal)
#         optim_grid.zero_grad()
#         loss.backward()
#         optim_grid.step()  
#         if iter == num_steps -1:
#             with torch.no_grad():
#                 output = grid(x)
#                 loss = loss_fn(output, signal)
#                 loss_val = loss_fn(output, signal).cpu().squeeze()
#                 psnr = -10*torch.log10(loss_val).numpy()
#                 grid_psnr = psnr
    
#     torch.cuda.empty_cache()
#     print(output.shape)
#     output2save = output.reshape(signal_shape).squeeze().detach().cpu().numpy()
#     print(output2save.shape)
#     plt.figure()
#     output2save = np.clip(output2save, 0, 1)
#     # output2save = (output2save - output2save.min()) / (output2save.max() - output2save.min())
#     plt.imshow(output2save)
#     plt.title(f'PSNR: {psnr}')
#     plt.tight_layout()
#     # plt.savefig(f'{model_name}_{bandlimit}.png')
#     plt.savefig(f'{max_param:0.0e}.png')
#     plt.close()

