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
import math
from torch.utils.checkpoint import checkpoint
from gsplat import rasterization, rasterization_2dgs
import cv2
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
class GSplat(nn.Module):
    def __init__(self, dimension, max_params, out_features=1, resolution = None, d= 1, model_type = '3dgs', device = 'cuda'):
        super().__init__()
        self.dimension = dimension
        self.resolution = resolution
        self.out_feature = out_features
        self.device = device
        if dimension == 2:
            self.d = self.out_feature
            total_params = 11 + self.d
            self.num_params = max_params
            self.params_list = {1e4:1e4//total_params, 3e4:3e4//total_params, 1e5:1e5//total_params, 3e5:3e5//total_params, 1e6:1e6//total_params, 3e6:3e6//total_params}
            if model_type == "3dgs":
                self.rasterize_fnc = rasterization
            elif model_type == "2dgs":
                self.rasterize_fnc = rasterization_2dgs
        self.num_points = int(self.params_list[max_params])
        self._init_gaussians(dimension)
        self.batch_training = False
        self.batch_size = 910
    def _init_gaussians(self, dimension = 2):
        if dimension == 2:
            bd = 2
            self.means = nn.Parameter(bd * (torch.rand(self.num_points, 3, device = self.device) - 0.5))
            self.scales = nn.Parameter(torch.rand(self.num_points, 3, device = self.device))
            # d = 3
            self.rgbs = nn.Parameter(torch.rand(self.num_points, self.d, device = self.device))

            u = torch.rand(self.num_points, 1, device = self.device)
            v = torch.rand(self.num_points, 1, device = self.device)
            w = torch.rand(self.num_points, 1, device = self.device)

            self.quats = nn.Parameter(torch.cat(
                [
                    torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                    torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                    torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                    torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
                ],
                -1,
            ))
            self.opacities = nn.Parameter(torch.ones((self.num_points), device = self.device))

            self.viewmat = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 8.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                device = self.device
            )
            self.background = torch.zeros(self.d, device = self.device)

            self.viewmat.requires_grad = False

    def get_state_dict(self, dimension = 2):
        """Return current model parameters in a dict."""
        if dimension == 2:
            return {
                'means': self.means.cpu(),
                'scales': self.scales.cpu(),
                'quats': self.quats.cpu(),
                'opacities': self.opacities.cpu(),
                'rgbs': self.rgbs.cpu(),
            }
    def quat2rot(self, q):
        """
        Convert quaternions to rotation matrices.
        q: tensor of shape (N,4) -> interpreted as (x, y, z, w)
        """
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.stack([
            torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
            torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
            torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
        ], dim=1)
        return R

    def forward(self, coords, gt_image = None):
        if self.dimension == 2:        
            fov_x = math.pi / 2.0
            
            if len(coords.squeeze().shape) == 2:
                self.batch_training = True
            H, W = self.resolution[0], self.resolution[1]
            
            focal = 0.5 * float(W) / math.tan(0.5 * fov_x)

            K = torch.tensor(
                [
                    [focal, 0, W / 2],
                    [0, focal, H / 2],
                    [0, 0, 1],
                ],
                device=self.device,
            )

            renders = self.rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                W,
                H,
                packed=False,
            )[0]

            out_img = renders[0].squeeze()
            if out_img.shape[-1] == 3:
                pass
            else:
                out_img = out_img.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
            coords_grid = coords.unsqueeze(0)  # Shape: (1, 1, N, 2)
            
            # Bilinear interpolation to resample values at `coords`
            sampled_values = out_img
            if self.batch_training == True:
                if out_img.shape[1] == 3:
                    sampled_values = sampled_values.reshape([-1, self.out_feature])
                else:
                    sampled_values = sampled_values.reshape([-1, self.out_feature])
            else:
                if out_img.shape[1] == 3:
                    sampled_values = sampled_values.squeeze().permute(1,2,0)
                    
                else: pass
            return sampled_values  # [H, W, 1 or 3]

        elif self.dimension == 3:
            raise NotImplementedError("3D signal fitting is not supported in this implementation.")
  
