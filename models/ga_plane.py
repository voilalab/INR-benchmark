## Common functions and imports
import numpy as np
from PIL import Image
import skimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# from model_utils import *

interpolation = 'bilinear'
# interpolation = 'nearest'
bias = True
include_lowres = True

resolutions1 = [32, 64, 128, 192, 256, 320, 384, 448, 512]
resolutions2 = [1]
feature_dims = [32, 64, 128, 192, 256, 320, 384]
hidden_dims = [32, 64, 128, 256]
feature_dims = feature_dims[3]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# make a low resolution version of the image
def lowres(img, shrink_factor=None, model_size=None, resample=Image.BILINEAR):
  im = Image.fromarray(np.array(img*255, dtype=np.uint8))
  width, height = im.size
  if shrink_factor is None:  # autocompute the shrink factor to achieve a certain model size
    assert model_size is not None
    shrink_factor = int(np.sqrt(height * width / model_size))
  newsize = (width//shrink_factor, height//shrink_factor)
  if newsize[0] < 1 or newsize[1] < 1:
    return np.zeros(img.shape)
  im_small = im.resize(newsize, resample=resample)
  return np.array(im_small.resize((width, height), resample=resample)) / 255, newsize[0] * newsize[1]


# make a low rank approximation of the image
def lowrank(residual, rank_reduction=None, model_size=None):
  assert residual.shape[0] == residual.shape[1]
  if rank_reduction is None:  # autocompute the rank reduction to achieve a certain model size
    assert model_size is not None
    rank_reduction = 2*(residual.shape[0]**2) // model_size
  keep_rank = residual.shape[0]//rank_reduction
  if keep_rank < 1:
    return np.zeros_like(residual), 0
  U, S, Vh = np.linalg.svd(residual, full_matrices=True)
  S[keep_rank:] = 0
  smat = np.diag(S)
  low_rank = np.real(U @ smat @ Vh)
  return low_rank, keep_rank * residual.shape[0] * 2


# make a sparse approximation of the image
def sparse(residual, sparse_reduction=None, model_size=None):
  flat = residual.flatten()
  if sparse_reduction is None:  # autocompute s to achieve a certain model size
    s = model_size // 2
  else:
    s = len(flat) // (sparse_reduction**2)
  result = np.zeros_like(flat)
  idx = np.flip(np.argsort(residual.flatten()))[0:s] # no need to take absolute value since image values are all nonnegative
  result[idx] = flat[idx]
  return result.reshape(residual.shape), s * 2

def psnr_normalized(gt, recon):
  # First compute the normalization factor based on the gt image
  scale = 1
  if np.max(gt) > 1:
    scale = 255
  residual = (gt - recon) / scale
  mse = np.mean(np.square(residual))
  return -10*np.log10(mse)


# Filter a set of points to find those that are pareto optimal
# assuming higher is better for y and smaller is better for x
def find_pareto(xvals, yvals):
  # First sort according to xvals, in increasing order
  idx = np.argsort(xvals)
  xvals = xvals[idx]
  yvals = yvals[idx]
  # Calculate pareto frontier indices
  pareto_indices = []
  for i in range(len(xvals)):
    is_pareto = True
    for j in range(len(xvals)): # was 0.015
      if i != j and xvals[j] <= xvals[i] + 0.0 and yvals[j] >= yvals[i]:  # TODO: make this a little more restrictive
        is_pareto = False
        break
    if is_pareto:
      pareto_indices.append(i)
  return xvals[pareto_indices], yvals[pareto_indices], idx[pareto_indices]



class GAPlane(nn.Module):
    def __init__(self, dimension, max_params, out_features = 1, dim2 = 20, dim_features = feature_dims, m=0, resolution = [1000, 1000], operation='multiply', decoder = 'nonconvex', interpolation=interpolation, bias=bias, include_lowres=True, hidden_out = False):
        super(GAPlane, self).__init__()
        self.operation = operation
        self.decoder = decoder
        self.interpolation = interpolation
        self.bias = bias
        self.include_lowres = include_lowres
        self.out_features = out_features
        self.dimension = dimension
        self.num_hidden_layers = 1
        if self.dimension == 2: # separate x, y and apply those into the line feature x and y
          if len(resolution) == 3:
            self.resx, self.resy, _ = resolution
          else:
            self.resx, self.resy = resolution
          dim1 = self.resx
        else: 
          self.resx, self.resy, dim1 = resolution
        if out_features == 3:
          if self.resx > 1000: #super_res
            self.resx, self.resy = self.resx//4, self.resy//4
        if dimension == 2 and max_params == 1e4 and self.resx == 1000:
          dim2 = 11
          res_val = 550
          self.resx, self.resy = res_val, res_val
        a = self.num_hidden_layers
        c = out_features - max_params
        if dimension == 3:
          dim3 = 5
          enc_params = self.resx + self.resy + dim1 + (dim2**2)*3 + dim3**3
          b = enc_params + 1 + out_features
          width = (-b + np.sqrt(b*b - 4*a*c)) / (2*a) # formula of roots
          dim_features = int(np.round(width))

          tqdm.write(f'GAPlane number of features: {dim_features} dim2: {dim2} x {self.resx} y {self.resy} z {dim1} a {a} b {b} c {c}')
          # Define the feature tensors
          # torch.manual_seed(0)
          if operation == 'multiply':
            self.line_feature_x = nn.Parameter(torch.rand(dim_features, self.resx)*0.15 + 0.1)  
            self.line_feature_y = nn.Parameter(torch.rand(dim_features, self.resy)*0.15 + 0.1)
            self.line_feature_z = nn.Parameter(torch.rand(dim_features, dim1)*0.15 + 0.1)
          else:
            self.line_feature_x = nn.Parameter(torch.rand(dim_features, dim1)*0.03 + 0.005) 
            self.line_feature_y = nn.Parameter(torch.rand(dim_features, dim1)*0.03 + 0.005)
          if self.include_lowres:
            self.plane_feature = nn.Parameter(torch.randn(dim_features, dim2, dim2)*0.01)
            self.plane_feature_yz = nn.Parameter(torch.randn(dim_features, dim2, dim2)*0.01)
            self.plane_feature_zx = nn.Parameter(torch.randn(dim_features, dim2, dim2)*0.01)
          
          self.volume_feature = nn.Parameter(torch.randn(dim_features, dim3, dim3, dim3)*0.001)
        else: # when signal is 2 dimension
          enc_params = self.resx + self.resy + dim2**2
          b = enc_params + 1 + out_features
          width = (-b + np.sqrt(b*b - 4*a*c)) / (2*a) # formula of roots
          dim_features = int(np.round(width))
          
          tqdm.write(f'GAPlane number of features: {dim_features} dim2: {dim2} x {self.resx} y {self.resy}  a {a} b {b} c {c}')
          # Define the feature tensors
          # torch.manual_seed(0)
          if operation == 'multiply':
            self.line_feature_x = nn.Parameter(torch.rand(dim_features, self.resx)*0.15 + 0.1)  
            self.line_feature_y = nn.Parameter(torch.rand(dim_features, self.resy)*0.15 + 0.1)
          else:
            self.line_feature_x = nn.Parameter(torch.rand(dim_features, self.resx)*0.03 + 0.005) 
            self.line_feature_y = nn.Parameter(torch.rand(dim_features, self.resy)*0.03 + 0.005)
          if self.include_lowres:
            self.plane_feature = nn.Parameter(torch.randn(dim_features, dim2, dim2)*0.01)
          
        # Define the decoder
        if decoder == 'linear':
          self.mlp = nn.Sequential(
            nn.Linear(dim_features, out_features, bias=self.bias),
          )
        elif decoder == 'nonconvex':
          self.mlp = nn.Sequential(
              nn.Linear(dim_features, dim_features, bias=self.bias),
              nn.ReLU(),
              nn.Linear(dim_features, out_features, bias=self.bias)
          )
        elif decoder == 'convex':
          self.fc1 = nn.Linear(dim_features, m, bias=self.bias)
          self.fc2 = nn.Linear(dim_features, m, bias=self.bias)
          self.fc2.weight.requires_grad = False
          if self.bias:
            self.fc2.bias.requires_grad = False

        else:
          raise ValueError(f"Invalid decoder {decoder}; expected linear, nonconvex, or convex")          

    def forward(self, coords):
        # # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)  # [batchx, batchy, 1]
        y_coords = coords[..., 1].unsqueeze(-1)  # [batchx, batchy, 1]
        # print(x_coords.shape, y_coords.shape)
        if coords.shape[-1] == 3:
          z_coords = coords[..., 2].unsqueeze(-1)  # [batchx, batchy, 1]
        gridx = torch.cat((x_coords, x_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        gridy = torch.cat((y_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        if coords.shape[-1] == 3:
          gridz = torch.cat((z_coords, z_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        if len(gridx.shape) >= 4:
          pass
        else: 
          gridx = gridx.unsqueeze(0)
          gridy = gridy.unsqueeze(0)
          if coords.shape[-1] == 3:
            gridz = gridz.unsqueeze(0)
        # Interpolate line features using grid_sample
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)  # [1, dim_features, dim1, 1]
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)  # [1, dim_features, dim1, 1]
        if coords.shape[-1] == 3:
          line_features_z = self.line_feature_z.unsqueeze(0).unsqueeze(-1)  # [1, dim_features, dim1, 1]
        # Get the feature tensors for grid_sample
        feature_x = F.grid_sample(line_features_x, gridx, mode=self.interpolation, padding_mode='border', align_corners=True)  # [1, dim_features, batchx, batchy]
        feature_y = F.grid_sample(line_features_y, gridy, mode=self.interpolation, padding_mode='border', align_corners=True)  # [1, dim_features, batchx, batchy]
        if coords.shape[-1] == 3:
          feature_z = F.grid_sample(line_features_z, gridz, mode=self.interpolation, padding_mode='border', align_corners=True)  # [1, dim_features, batchx, batchy]

        # Prepare for 2D interpolation for the plane feature
        sampled_plane_features = 0
        sampled_plane_features_yz = 0
        sampled_plane_features_zx = 0
        if self.include_lowres:
          plane_features = self.plane_feature.unsqueeze(0)  # [1, dim_features, dim2, dim2]
          plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
          if coords.shape[-1] == 3:
            plane_features_yz = self.plane_feature_yz.unsqueeze(0)  # [1, dim_features, dim2, dim2]
            plane_grid_yz = torch.cat((y_coords, z_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
            plane_features_zx = self.plane_feature_zx.unsqueeze(0)  # [1, dim_features, dim2, dim2]
            plane_grid_zx = torch.cat((z_coords, x_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
          
          if len(plane_grid.shape) >= 4:
            pass
          else: 
            plane_grid = plane_grid.unsqueeze(0)
            if coords.shape[-1] == 3:
              plane_grid_yz = plane_grid_yz.unsqueeze(0)
              plane_grid_zx = plane_grid_zx.unsqueeze(0)
          
          # Sample from the plane feature using grid_sample
          sampled_plane_features = F.grid_sample(plane_features, plane_grid, mode=self.interpolation, align_corners=True)  # [1, dim_features, batchx, batchy]
          if coords.shape[-1] == 3:
            volume_features = self.volume_feature.unsqueeze(0)  # [1, dim_features, dim2, dim2]
            volume_grid = torch.cat((x_coords, y_coords, z_coords), dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, batchx, batchy, 2]
            sampled_plane_features_yz = F.grid_sample(plane_features_yz, plane_grid_yz, mode=self.interpolation, align_corners=True)  # [1, dim_features, batchx, batchy]
            sampled_plane_features_zx = F.grid_sample(plane_features_zx, plane_grid_zx, mode=self.interpolation, align_corners=True)  # [1, dim_features, batchx, batchy]
        
        sampled_volume_features = 0
        if coords.shape[-1] == 3:
          sampled_volume_features = F.grid_sample(volume_features, volume_grid, mode=self.interpolation, align_corners=True)  # [1, dim_features, batchx, batchy]
        # Combine features
        if self.operation == 'add':
            
            combined_features = feature_x + feature_y + sampled_plane_features + sampled_plane_features_yz + sampled_plane_features_zx + sampled_volume_features # [1, dim_features, batchx, batchy]
        elif self.operation == 'multiply':
            if coords.shape[-1] == 3:
              feature_x, feature_y, feature_z = feature_x.unsqueeze(2), feature_y.unsqueeze(2), feature_z.unsqueeze(2)
              sampled_plane_features, sampled_plane_features_yz, sampled_plane_features_zx = sampled_plane_features.unsqueeze(2), sampled_plane_features_yz.unsqueeze(2), sampled_plane_features_zx.unsqueeze(2)
              lines = feature_x * feature_y * feature_z
              planes = sampled_plane_features * feature_z + sampled_plane_features_yz * feature_x + sampled_plane_features_zx * feature_y

              combined_features = lines + planes + sampled_volume_features  # [1, dim_features, batchx, batchy]
            else: 
              combined_features = feature_x * feature_y + sampled_plane_features # [1, dim_features, batchx, batchy]
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or multiply")
        # Reorder axes so this can be fed to the MLP
        if coords.shape[-1] == 3:
          combined_features = combined_features.squeeze(0).permute(1, 2, 3, 0)  # [batchx, batchy, batchz, dim_features]
        else:
          combined_features = combined_features.squeeze(0).permute(1, 2, 0)  # [batchx, batchy, dim_features]

        # Pass through decoder
        if self.decoder == 'linear' or self.decoder == 'nonconvex':
          
          output = self.mlp(combined_features).squeeze()  # [batchx, batchy]
          
        else:  # convex
          output = self.fc1(combined_features) * (self.fc2(combined_features) > 0)  # [batchx, batchy, m]
          output = torch.mean(output, dim=-1)  # [batchx, batchy]
        
        return output


def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    dimension = 3
    max_params = 1e6
    sigma = 1
    for model_size in [1e4,3e4,1e5,3e5,1e6,3e6]:
        inr = GAPlane(dimension, max_params=model_size, resolution = [100,100,100])
        
        print(inr, count_parameters(inr))