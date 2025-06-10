import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import os
import subprocess
import random
from bandlimited_signal import *
import os
import sys
import glob
import pdb
import numpy as np
from scipy import signal
import torch
from torch import nn
import kornia
import cv2
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Lambda
import skimage
from PIL import Image
from torch.utils.data import Dataset

def radon(imten, angles, is_3d=False):
    '''
        Compute forward radon operation
        
        Inputs:
            imten: (1, nimg, H, W) image tensor
            angles: (nangles) angles tensor -- should be on same device as 
                imten
        Outputs:
            sinogram: (nimg, nangles, W) sinogram
    '''
    nangles = len(angles)
    imten_rep = torch.repeat_interleave(imten, nangles, 0)
    
    imten_rot = kornia.geometry.rotate(imten_rep, angles)
    
    if is_3d:
        sinogram = imten_rot.sum(2).squeeze().permute(1, 0, 2)
    else:
        sinogram = imten_rot.sum(2).squeeze()
        
    return sinogram

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def fft_custom(signal):
    F_signal = torch.fft.fftn(signal)
    F_signal = torch.fft.fftshift(F_signal)
    return F_signal

def ifft_custom(signal):
    signal = torch.fft.ifftshift(signal)
    F_signal = torch.fft.ifftn(signal)
    return F_signal

def print_memory_usage(device=None, tag=""):
    """Print GPU and CPU memory usage for a specific device."""

    if torch.cuda.is_available():
        # Check if a specific device is provided
        if device is None:
            device = torch.cuda.current_device()  # Default to the current device
        elif isinstance(device, torch.device):
            device = device.index  # Extract the index if it's a torch.device object

        gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # In MB
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # In MB
        print(f"{tag} GPU {device} Memory - Allocated: {gpu_memory_allocated:.2f} MB, Reserved: {gpu_memory_reserved:.2f} MB")
    else:
        print(f"{tag} No GPU available.")

def get_free_gpu(cuda_num = None):
    # get GPU memory information using nvidia-smi
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
        encoding='utf-8')
    
    # Check free memory spaces
    memory_free = [int(x) for x in result.strip().split('\n')]
    if cuda_num != None:
        return memory_free[cuda_num]
    # Pick the least used one
    rev_idx = int(torch.argmax(torch.flip(torch.tensor(memory_free), dims=[0])))
    best_gpu = len(memory_free) - 1 - rev_idx
    
    for i, free_mem in enumerate(memory_free):
        print(f'GPU {i} : {free_mem}MB', end = '\t')

    tqdm.write(f"\nUsing GPU: {best_gpu} with {memory_free[best_gpu]} MB free memory.")
    return best_gpu

'''
make the folders to save figures
'''
def make_folders(folder_name):
    
    # os.makedirs(f'band_limit_figs/{formatted_time}')
    os.makedirs(folder_name, exist_ok=True)
    if folder_name[-1] != '/':
        folder_name += '/' 
    target_dir = folder_name
    return target_dir

'''
visualize the target and predicted output and save it in the folder
'''
def viz_compare(signal, output, model, bandlimit, max_params, psnr, dimension, target_dir, visualize = False):
    model_name = model.__class__.__name__
    signal_name = signal.__class__.__name__
    if visualize == True:
        plt.figure()
        plt.subplot(1,2,1)
        if dimension == 2:
            plt.imshow(signal.signal)
            plt.clim([0,1])
            plt.colorbar(fraction=0.046, pad=0.04)
        elif dimension == 3:
            ax = plt.axes(projection='3d')
            ax.plot_surface(signal.signal[0], signal.signal[1], signal.signal[2], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
            
        plt.title('target signal')
        plt.axis('off')

        plt.subplot(1,2,2)
        if dimension == 2:
            plt.imshow(output)
            plt.clim([0,1])
            plt.colorbar(fraction=0.046, pad=0.04)
        elif dimension == 3:
            ax = plt.axes(projection='3d')
            ax.plot_surface(output[0], output[1], output[2], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
        
        plt.axis('off')
        plt.title(model_name+f' PSNR : {psnr:.2f}')
        # plt.imshow(np.concatenate([signal.signal, output], axis = 1))
        plt.tight_layout()
        plt.savefig(target_dir+f'/{bandlimit}/{signal_name}_bandlimit_{bandlimit}_size_{max_params:1.0e}_{model_name}.png')
        plt.close()
        tqdm.write(f'for bandlimit {bandlimit}, {model_name} psnr is {psnr:.2f}')

        # for saving only target figure
        plt.figure()
        plt.imshow(output)
        plt.clim([0,1])
        plt.title(model_name+f' PSNR : {psnr:.2f}')
        # plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(target_dir+f'/{bandlimit}/only_targets/{signal_name}_bandlimit_{bandlimit}_size_{max_params:1.0e}_{model_name}.png')
        plt.close()

        # for saving GT
        GT_path = target_dir+f'/{bandlimit}/only_targets/{signal_name}_bandlimit_{bandlimit}_size_{max_params:1.0e}_GT.png'
        if not os.path.exists(GT_path):
            plt.figure()
            plt.imshow(signal.signal)
            plt.title(signal_name)
            # plt.colorbar(fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(GT_path)
            plt.close()
    else: pass

def PSNR(output, signal):
    mse = np.mean(np.square(signal.signal - output))
    psnr = -10 * np.log10(mse)
    return psnr

def find_best_PSNR(psnr, best_psnr, omega, sigma):
    if psnr > best_psnr:
        best_psnr = psnr
        best_omega = 0
        best_sigma = sigma
    return best_psnr, best_omega, best_sigma

def TV_Reg(grid):
    """
    Compute 2D Total Variation for a given grid with L2 norm for inner and average over pixels.
    Args:
        grid (torch.Tensor): [C, H, W] or [H, W] input tensor
    Returns:
        tv (torch.Tensor): Total variation value
    """
    # Compute horizontal and vertical differences
    tv_h = (grid[:, 1:, :] - grid[:, :-1, :]) ** 2  # Horizontal difference squared
    tv_w = (grid[:, :, 1:] - grid[:, :, :-1]) ** 2  # Vertical difference squared

    # Combine horizontal and vertical differences with L2 norm (sqrt of sum of squares)
    tv_inner = torch.sqrt(torch.clamp(tv_h[:, :, :-1] + tv_w[:, :-1, :], min=1e-6))

    # Average over all pixels
    tv = tv_inner.mean()
    # print(f'TV h {tv_h}, TV w {tv_w} TV_inner {tv_inner} TV_inner mean {tv}')

    return tv


# Save the current sigma's data
def save_scores(model_name, psnrs, bandlimit, max_params, sigma = 0, omega= 0 , grid_psnr = 0, psnr = 0):
    psnrs['bandlimits'].append(bandlimit)
    psnrs['num_params'].append(f'{max_params:1.0e}')
    psnrs['sigma'].append(sigma)
    psnrs['omega'].append(omega)
    psnrs['grid'].append(f'{grid_psnr:.2f}')
    psnrs[model_name].append(f'{psnr:2f}')
    return psnrs

def find_folder(target_folder_name, search_directory):
    """
    Function to search for folders whose names contain the specified substring in the top-level directory only.
    
    Args:
    - target_folder_name (str): The substring of the folder name to search for
    - search_directory (str): The path of the top-level directory to start searching in
    
    Returns:
    - list: A list of paths to the found folders
    """
    found_folders = []
    
    # Use os.listdir to scan only the current directory (no recursion)
    for item in os.listdir(search_directory):
        item_path = os.path.join(search_directory, item)
        # Check if it's a directory and contains the target folder name
        if os.path.isdir(item_path) and target_folder_name in item:
            found_folders.append(item_path)
    
    return found_folders

def set_colormap(n = 7):
    colors = plt.cm.plasma(np.linspace(0,1,n))
    return colors

def plot_colormap(model_name = None):
    colors = {'FourFeat' : "#56B4E9",
              'Siren' : "#D55E00",
              'Wire' : "#27ae60",
              'Instant_NGP' : "#EA3680",
              'GAPlane' : "#9a0eea",
              'GSplat' : '#f1c40f',
              'Grid' : "#363737",
              'BACON': '#0e6251'}
    
    
    if 'FourFeat' in model_name:
        model_color = 'FourFeat'
    elif 'Grid' in model_name:
        model_color = 'Grid'
    else: model_color = model_name
    return str(colors.get(model_color, "#000000"))

def plot_linestyle(model_name = None):    
    if 'FourFeat_Embed' == model_name:
        linestyle = '-.'
    elif 'FourFeat_5e3' == model_name:
        linestyle = '--'
    elif 'Grid_TV' == model_name:
        linestyle = '--'
    else:
        linestyle = '-'
    return linestyle

def plot_marker(model_name = None):
    if model_name == 'Siren':
        marker = 'o'
    elif model_name == 'Wire':
        marker = 's'
    elif model_name == 'Instant_NGP':
        marker = '^'
    elif model_name == 'GAPlane':
        marker = 'D'
    else: marker = None
    return marker

def plot_model_name(model_name = None):
    if 'FourFeat' == model_name:
        model_output = r'FFN'
    elif 'FourFeat_Embed' == model_name:
        model_output = r'FFN Blend'
    elif 'FourFeat_5e3' == model_name:
        model_output = r'FFN $(\sigma = 5000)$'
    elif 'Siren' == model_name:
        model_output = r'SIREN'
    elif 'Wire' == model_name:
        model_output = r'WIRE'
    elif 'Instant_NGP' == model_name:
        model_output = r'Instant-NGP'
    elif 'GAPlane' == model_name:
        model_output = r'GA-Planes'
    elif 'Grid' in  model_name:
        model_output = r'Grid'
    elif 'Gaussian' in model_name:
        model_output = r'GSplat'
    elif 'BACON' in model_name:
        model_output = r'BACON'
    return model_output

def plot_line_width(model_name =None):
    if 'FourFeat' in model_name:
        model_output = 2.5
    elif 'Grid' in model_name:
        model_output = 2
    else: model_output = 1.5
    return model_output

def zoom_output(image, save_path, zoom_ratio = 0.35, window_ratio = 0.35):
    H, W = image.shape[:2]

    ratio = zoom_ratio
    small_window_ratio = window_ratio
    zoom_regions = [(230, 110, int(ratio*W), int(ratio*H))]*6

    for idx in range(6):
        fig, ax = plt.subplots(figsize=(5, 5))  
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        x, y, width, height = zoom_regions[idx]
        zoomed_image = image[y:y+height, x:x+width]
        
        ax.imshow(zoomed_image)
        ax.axis('off') 
        
        inset_ax = ax.inset_axes([0.65, 0.00, small_window_ratio, small_window_ratio])
        inset_ax.imshow(image)
        inset_ax.axis('off')
        
        border = patches.Rectangle(
            (0, 0), 1, 1,
            transform=inset_ax.transAxes,
            linewidth=2, 
            edgecolor="#e0d291", 
            facecolor="none"
        )
        inset_ax.add_patch(border)
        
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        inset_ax.add_patch(rect)
        
        plt.savefig(f'{save_path.jpeg}', bbox_inches='tight', pad_inches = 0)

# set global seed
def set_seed(seed):
    tqdm.write(f"Selected Seed: {seed}")
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if CUDA

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def get_mgrid(sidelen, dim=2, centered=True, include_end=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if include_end:
        denom = [s-1 for s in sidelen]
    else:
        denom = sidelen

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / denom[0]
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / denom[1]
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / denom[0]
        pixel_coords[..., 1] = pixel_coords[..., 1] / denom[1]
        pixel_coords[..., 2] = pixel_coords[..., 2] / denom[2]
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    if centered:
        pixel_coords -= 0.5

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

class ImageFile(Dataset):
    def __init__(self, image, grayscale=False, resolution=None,
                 root_path=None, crop_square=True, url=None):

        super().__init__()

        # image: torch.Tensor, shape like (C, H, W)
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if image.shape[0] == 1 or image.shape[0] == 3:
            # Convert from (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
        
        # Clamp and convert to uint8 if it's float
        if image.dtype in [np.float32, np.float64]:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)

        self.img = Image.fromarray(image).resize(resolution)
        self.img_channels = len(self.img.mode)
        self.resolution = self.img.size

        if crop_square:  # preserve aspect ratio
            self.img = crop_max_square(self.img)

        if resolution is not None:
            self.resolution = resolution
            self.img = self.img.resize(resolution, Image.LANCZOS)

        self.img = np.array(self.img)
        self.img = self.img.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img
    
class ImageWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, compute_diff='all', centered=True,
                 include_end=False, multiscale=False, stages=3):

        self.compute_diff = compute_diff
        self.centered = centered
        self.include_end = include_end
        self.transform = Compose([
            ToTensor(),
        ])

        self.dataset = dataset
        self.mgrid = get_mgrid(self.dataset.resolution, centered=centered, include_end=include_end)

        # sample pixel centers
        self.mgrid = self.mgrid + 1 / (2 * self.dataset.resolution[0])
        self.radii = 1 / self.dataset.resolution[0] * 2/np.sqrt(12)
        self.radii = [(self.radii * 2**i).astype(np.float32) for i in range(3)]
        self.radii.reverse()

        img = self.transform(self.dataset[0])
        # print('inside wrapper', img.shape)
        _, self.rows, self.cols = img.shape

        self.img_chw = img
        self.img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)
        
        self.imgs = []
        self.multiscale = multiscale
        img = img.permute(1, 2, 0).numpy()
        for i in range(stages):
            tmp = skimage.transform.resize(img, [s//2**i for s in (self.rows, self.cols)])
            tmp = skimage.transform.resize(tmp, (self.rows, self.cols))
            self.imgs.append(torch.from_numpy(tmp).view(-1, self.dataset.img_channels))
        self.imgs.reverse()
        # print('after wrapper', img.shape)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        coords = self.mgrid
        img = self.img
        in_dict = {'coords': coords, 'radii': self.radii}
        gt_dict = {'img': img}

        if self.multiscale:
            gt_dict['img'] = self.imgs

        return in_dict, gt_dict

def init_dataloader(img, batch_size=1, res=[1000, 1000]):
    ''' load image or voxel datasets, dataloader '''
    
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    ndim = img.ndim

    # ---------------- 2D CASE ----------------
    if ndim == 2 or (ndim == 3 and img.shape[0] in [1, 3]):  # (H,W) or (C,H,W)
        trn_dataset = ImageFile(img, resolution=(res[0], res[1]))
        trn_dataset = ImageWrapper(trn_dataset, centered=True,
                                   include_end=False,
                                   multiscale=False,
                                   stages=3)

    # ---------------- 3D CASE ----------------
    elif ndim == 3 or (ndim == 4 and img.shape[0] == 1):  # (D,H,W) or (C,D,H,W)
        class VoxelDataset(torch.utils.data.Dataset):
            def __init__(self, voxel, resolution=(100, 100, 100)):
                super().__init__()
                if voxel.ndim == 4:
                    voxel = voxel[0]
                self.voxel = voxel.astype(np.float32)
                self.resolution = self.voxel.shape

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return self.voxel

        class VoxelWrapper(torch.utils.data.Dataset):
            def __init__(self, dataset, centered=True, include_end=False):
                self.dataset = dataset
                self.voxel = dataset[0]  # shape: (D, H, W)
                self.centered = centered
                self.include_end = include_end

                self.resolution = self.voxel.shape
                self.mgrid = get_mgrid(self.resolution, dim=3,
                                       centered=centered, include_end=include_end)  # (N, 3)
                self.gt = torch.from_numpy(self.voxel).reshape(-1, 1)  # flatten to (N, 1)

            def __len__(self):
                return 1

            def __getitem__(self, idx):
                return {'coords': self.mgrid}, {'img': self.gt}

        trn_dataset = VoxelDataset(img)
        trn_dataset = VoxelWrapper(trn_dataset)

    else:
        raise ValueError(f"Unsupported img shape: {img.shape}")

    dataloader = DataLoader(trn_dataset, shuffle=True,
                            batch_size=batch_size,
                            pin_memory=True,
                            num_workers=0)

    return trn_dataset, dataloader

def multiscale_image_mse(model_output, gt, use_resized=False):
    if use_resized:
        loss = [(out - gt_img)**2 for out, gt_img in zip(model_output['model_out']['output'], gt['img'])]
    else:
        loss = [(out - gt)**2 for out in model_output]

    loss = torch.stack(loss).mean()

    return {'func_loss': loss}

def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cuda(value)})
        elif isinstance(value, list) or isinstance(value, tuple):
            if isinstance(value[0], torch.Tensor):
                tmp.update({key: [v.cuda() for v in value]})
        else:
            tmp.update({key: value})
    return tmp

def compute_iou(pred: torch.Tensor, gt: torch.Tensor, device='cuda', threshold: float = 0.2) -> float:
    """
    Computes IoU for 3D voxel grids using PyTorch, with thresholding for float inputs.

    Args:
        pred (torch.Tensor or np.ndarray): Predicted voxel grid (float or binary), shape: (D, H, W)
        gt (torch.Tensor or np.ndarray): Ground truth voxel grid (float or binary), shape: (D, H, W)
        threshold (float): Threshold to binarize the inputs
        device (str): 'cuda' or 'cpu'

    Returns:
        float: IoU score
    """
    assert pred.shape == gt.shape, "Shape mismatch"

    # Convert from numpy to torch if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)

    # Threshold before bool casting
    pred = (pred > threshold).bool().to(device)
    gt = (gt > threshold).bool().to(device)

    intersection = torch.logical_and(pred, gt).sum().float()
    union = torch.logical_or(pred, gt).sum().float()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()

def preprocess_for_lpips(img, device='cuda'):
    # If input is numpy and in HWC format, transpose to CHW
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            # Grayscale → (1, H, W) → repeat to (3, H, W)
            img = np.stack([img]*3, axis=0)
        elif img.ndim == 3 and img.shape[2] == 3:
            # HWC → CHW
            img = np.transpose(img, (2, 0, 1))
        elif img.ndim == 3 and img.shape[0] == 1:
            # (1, H, W) → (3, H, W)
            img = np.repeat(img, 3, axis=0)

    img = torch.tensor(img).float().to(device)

    # Normalize to [0, 1] if it’s in 0~255
    if img.max() > 1.0:
        img = img / 255.0

    # LPIPS expects input shape (N, 3, H, W) and range [-1, 1]
    img = img.unsqueeze(0)  # Add batch dimension
    img = img * 2 - 1

    return img


def compute_lpips(gt, pred, device = 'cuda'):
    gt_lpips = preprocess_for_lpips(gt, device = device)
    pred_lpips = preprocess_for_lpips(pred, device = device)
    with torch.no_grad():
        score = lpips_fn(gt_lpips, pred_lpips).to(device)
    return score.item()
