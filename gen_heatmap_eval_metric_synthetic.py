import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bandlimited_signal import *
from matplotlib.colors import Normalize
from utils import *
import matplotlib.gridspec as gridspec
from matplotlib import font_manager as fm
import lpips
from skimage.metrics import structural_similarity as ssim
import pickle

device = 'cuda:0'

def find_folder(target_folder_name, search_directory, iters=1000):
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
        if os.path.isdir(item_path) and target_folder_name in item and str(iters) in item:
            found_folders.append(item_path)
    
    return found_folders

def eval_dataFrame(df_list, model_name, iters = False):
    max_df = df_list[0].copy()  # Start with a copy of the first DataFrame structure
    min_df = max_df.copy()
    mean_df = sum(df_list) / len(df_list)

    concat_df = pd.concat(df_list, axis=0)
    max_df = concat_df.groupby(concat_df.index).max()
    min_df = concat_df.groupby(concat_df.index).min()
    median_df = concat_df.groupby(concat_df.index).median()
    result_df = {'max':max_df, 'mean':mean_df, 'min':min_df, 'median':median_df}
     
    return result_df

def set_colormap(n = 7):
    colors = plt.cm.plasma(np.linspace(0,1,n))
    return colors

def PSNR(output, signal):
    mse = np.mean(np.square(signal.signal - output))
    psnr = -10 * np.log10(mse)
    return psnr

def mask_generator(img_size = 1000, visualize = False, dst_edge = 20):
    center = (img_size // 2, img_size // 2)  # Center of the circle
    radius = img_size // 2 - dst_edge # Maximum radius
    rads = np.linspace(0, radius, 10)  # Generate 10 intervals from 0 to radius

    # Generate coordinate grid
    y, x = np.ogrid[:img_size, :img_size]
    dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Create 9 ring masks
    masks = []
    for i in range(1, len(rads)):  # Create a ring for each interval
        inner_radius = rads[i-1]
        outer_radius = rads[i]
        
        # Create a mask where inner_radius <= dist_from_center < outer_radius
        mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
        masks.append(mask)
        
        if visualize:
            # Visualize the mask applied to the image (optional)
            plt.imshow(mask, cmap='gray')
            plt.title(f'Ring {i}')
            plt.show()

    masks = np.stack(masks, axis = 0)
    return masks
def PSNR_star(signal, target, multi_dim = False):
    if multi_dim:
        mse = np.mean(np.square(signal - target), axis=(-2,-1))
        psnr = -10 * np.log10(mse)
    else:    
        mse = np.mean(np.square(signal - target))
        psnr = -10 * np.log10(mse)
    return psnr

lpips_fn = lpips.LPIPS(net='vgg').to(device)

def preprocess_for_lpips(img, device = 'cuda'):
    if img.ndim == 2:  # grayscale -> 3 channel
        img = np.stack([img]*3, axis=0)
    elif img.shape[0] == 1:  # (1, H, W) -> (3, H, W)
        img = np.repeat(img, 3, axis=0)

    img = torch.tensor(img).float().to(device)
    img = img.unsqueeze(0)  # (1, 3, H, W)
    img = img * 2 - 1  # [0,1] -> [-1,1]
    return img

def compute_lpips(gt, pred, device = 'cuda'):
    gt_lpips = preprocess_for_lpips(gt, device = device)
    pred_lpips = preprocess_for_lpips(pred, device = device)
    with torch.no_grad():
        score = lpips_fn(gt_lpips, pred_lpips).to(device)
    return score.item()

def to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj
     
task_names = ['2d_SparseSphereSignal', '2d_BandlimitedSignal', '3d_SparseSphereSignal', '3d_BandlimitedSignal', '2d_Sierpinski', '2d_StarTarget']#, '3d_VoxelFitting']
task_names_vis = ['2D Spheres Signal', '2D Bandlimited Signal', '3D Spheres Signal', '3D Bandlimited Signal', '2D Sierpinski Signal', '2D Star Target Signal']

signals = [SparseSphereSignal, BandlimitedSignal, Sierpinski, StarTarget, Voxel_Fitting]
model_sizes = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
model_sizes_label = ['1e4', '3e4', '1e5', '3e5', '1e6', '3e6']
model_sizes_exp = [r'$1\times10^4$',r'$3\times10^4$', r'$1\times10^5$',r'$3\times10^5$',r'$1\times10^6$',r'$3\times10^6$']
bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bandlimits_vis = ['.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']
seeds = ['1234', '5678', '2024', '7618', '7890']
label_font_size = 24
times_new_roman = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf', size=label_font_size)
times_new_roman_task = fm.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf', size=28)

for task_idx, task_name in enumerate(task_names):
    
    for i, signal in enumerate(signals):
        if task_name.split("_")[-1] == signal.__name__:
            break
    search_directory = f"band_limit_figs/{task_name}"  # Directory path to search in
    target_folder_name = "complete"  # Name of the folder to search for
    if not os.path.exists(f'{search_directory}/heatmaps'):
        os.makedirs(f'{search_directory}/heatmaps')
    models = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'GSplat', 'BACON', 'Grid']
    models2plot_2d = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'GSplat', 'BACON', 'Grid']
    models2plot_3d = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'BACON', 'Grid']
    if task_name[0] == '3':
        models2plot = models2plot_3d
    else:
        models2plot = models2plot_2d
    colors = set_colormap(len(models))
    num_bandwidth = 9
    if task_name == '3d_VoxelFitting':
        num_bandwidth = 1    
        bandlimits = [0.1]
        bandlimits_vis = ['.1']
    else:
        num_bandwidth = 9   
        bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        bandlimits_vis = ['.1', '.2', '.3', '.4', '.5', '.6', '.7', '.8', '.9']

    num_params = 6
    if task_name == '2d_StarTarget':
        results = {}
        for model_name in models:
            if '3d' in task_name and model_name == 'BACON':
                continue

            cnt = 0
            results[model_name] = {}
            for dirs, _, files in os.walk(search_directory):
                if len(files) != 0:
                    for file in files:
                        if ".npy" in file and model_name in file and 'BMat' not in file:
                            if model_name in file:
                                current_model_name = model_name
                            else:
                                continue  # Skip files that don't match any model
                            if current_model_name == model_name:
                                pass
                            else: continue
                            seed_val, params_val = file[-8:-4], file[-18:-13]
                            cnt = cnt + 1
                            if params_val in model_sizes:
                                # If the key already exists, append the new result
                                if params_val not in results[model_name]:
                                    results[model_name][params_val] = {}
                                results[model_name][params_val][seed_val] = np.load(dirs+'/'+file)
                            results[model_name]['time'] = pd.read_csv(dirs+'/times.csv')

        target = np.load('target_signals/star_resolution_target_40_1000.npy')                    
        targets = np.stack([target]*9,axis= 0)
        masks = mask_generator()

        outputs = {}
        for model_name in models:
            outputs[model_name] = []
            
            for idx, model_size in enumerate(model_sizes):
                for i, seed_val in enumerate(seeds):
                    current_signal = results[model_name][model_size][seed_val]
                    if i == 0:
                        temp = current_signal
                    else: temp += current_signal
                temp /= len(seeds)
                temp = np.clip(temp, 0, 1)
                
                outputs[model_name].append(temp)

            outputs[model_name] = np.stack([outputs[model_name]]*9, axis = 1)

        psnrs = {}
        for model_name in models:
            psnrs[model_name] = {}
            for model_size_idx in range(len(outputs[model_name])):
                psnrs[model_name][model_size_idx] = []
                for i in range(len(bandlimits)):
                    masked_signal = outputs[model_name][model_size_idx][i][masks[i]]
                    
                    psnrs[model_name][model_size_idx].append(PSNR_star(target[masks[i]], masked_signal))
    else:
        mean_med = 'median'
        view_all = True
        target_folders = {}
        psnrs = {}
        if not task_name[0] == '3':
            ssims = {}
            lpipses = {}
            ssims_std = {}
            lpipses_std = {}
            ssim_mean = {}
            lpips_mean = {}
        else:
            ious = {} 
            iou_mean = {} 
            iou_std = {} 
        psnr_std = {}
        psnr_mean = {}
        for model_name in models:

            print(model_name)
            psnrs[model_name] = {}
            if not task_name[0] == '3':
                ssims[model_name] = {}
                lpipses[model_name] = {}
                ssims_std[model_name] = {}
                lpipses_std[model_name] = {}
                lpips_mean[model_name] = {}
                ssim_mean[model_name] = {}
            else:
                ious[model_name] = {} 
                iou_mean[model_name] = {} 
                iou_std[model_name] = {} 
            psnr_std[model_name] = {}
            psnr_mean[model_name] = {}
            found_folders = find_folder(target_folder_name, search_directory)
            target_folders[model_name] = []
            df = []

            for folder_name in found_folders:
                if model_name in folder_name:
                    if model_name == 'FourFeat':
                        if 'FourFeat_Embed' not in folder_name and 'FourFeat_5e3' not in folder_name and 'FourFeat_Triple' not in folder_name:
                            target_folders[model_name].append(folder_name)
                            df.append(pd.read_csv(folder_name + '/records.csv'))
                    else: 
                        target_folders[model_name].append(folder_name)
                        df.append(pd.read_csv(folder_name + '/records.csv'))
            else:
                pass
            dfs = eval_dataFrame(df, model_name)
            print(model_name, target_folders[model_name])
            outputs = {}
            
            if task_name in ['2d_Sierpinski', '2d_StarTarget', '3d_VoxelFitting']:
                seeds = ['1234']
                model_sizes = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
            elif model_name == 'GSplat' and '3d' in task_name:
                continue
            else:  
                seeds = ['1234', '5678', '2024', '7618', '7890']
                model_sizes = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
                
            for model_size in model_sizes:
                outputs[model_size] = {}
                psnrs[model_name][model_size] = []
                if not task_name[0] == '3':
                    ssims[model_name][model_size] = []
                    lpipses[model_name][model_size] = []
                    ssims_std[model_name][model_size] = []
                    lpipses_std[model_name][model_size] = []
                    ssim_mean[model_name][model_size] = []
                    lpips_mean[model_name][model_size] = []
                else:
                    ious[model_name][model_size] = []
                    iou_mean[model_name][model_size] = []
                    iou_std[model_name][model_size] = []
                psnr_std[model_name][model_size] = []
                psnr_mean[model_name][model_size] = []
                
                for bandlimit_idx, bandlimit in enumerate(bandlimits):
                    outputs[model_size][str(bandlimit)] = {}
                    psnrs[model_name][model_size].append([])
                    psnr_std[model_name][model_size].append([])
                    psnr_mean[model_name][model_size].append([])
                    if not task_name[0] == '3':
                        ssims[model_name][model_size].append([])
                        lpipses[model_name][model_size].append([])
                        ssims_std[model_name][model_size].append([])
                        lpipses_std[model_name][model_size].append([])
                        ssim_mean[model_name][model_size].append([])
                        lpips_mean[model_name][model_size].append([])
                    else:
                        ious[model_name][model_size].append([])
                        iou_mean[model_name][model_size].append([])
                        iou_std[model_name][model_size].append([])
                    for i, seed in enumerate(seeds):
                        outputs[model_size][str(bandlimit)][target_folders[model_name][i].split("_")[-2]] = np.load(f'{target_folders[model_name][i]}/{model_name}_output_{model_size}_{bandlimit}_{int(target_folders[model_name][i].split("_")[-2])}.npy')
                    for i, seed in enumerate(seeds):
                        target_signal = signal(dimension = int(task_name.split('_')[0][0]), length = 1000, bandlimit = bandlimit, seed = int(seed))
                        output2eval = outputs[model_size][str(bandlimit)][seed]
                        psnrs[model_name][model_size][bandlimit_idx].append(PSNR(output2eval, target_signal))
                        if not task_name[0] == '3':
                            ssim_val = ssim(output2eval, target_signal.signal,  data_range=target_signal.signal.max() - target_signal.signal.min())
                            lpips_val = compute_lpips(output2eval, target_signal.signal, device)
                            ssims[model_name][model_size][bandlimit_idx].append(ssim_val)
                            lpipses[model_name][model_size][bandlimit_idx].append(lpips_val)
                        else: 
                            iou_val = compute_iou(output2eval, target_signal.signal, device)
                            ious[model_name][model_size][bandlimit_idx].append(iou_val)

                    psnr_std[model_name][model_size][bandlimit_idx] = np.std(psnrs[model_name][model_size][bandlimit_idx])
                    psnr_mean[model_name][model_size][bandlimit_idx] = np.mean(psnrs[model_name][model_size][bandlimit_idx])
                    if not task_name[0] == '3':
                        ssim_mean[model_name][model_size][bandlimit_idx] = np.mean(ssims[model_name][model_size][bandlimit_idx])
                        lpips_mean[model_name][model_size][bandlimit_idx] = np.mean(lpipses[model_name][model_size][bandlimit_idx])
                        ssims_std[model_name][model_size][bandlimit_idx] = np.std(ssims[model_name][model_size][bandlimit_idx])
                        lpipses_std[model_name][model_size][bandlimit_idx] = np.std(lpipses[model_name][model_size][bandlimit_idx])
                    else:
                        iou_mean[model_name][model_size][bandlimit_idx] = np.mean(ious[model_name][model_size][bandlimit_idx])
                        iou_std[model_name][model_size][bandlimit_idx] = np.std(ious[model_name][model_size][bandlimit_idx])
                psnr_std[model_name][model_size] = np.array(psnr_std[model_name][model_size])
                psnr_mean[model_name][model_size] = np.array(psnr_mean[model_name][model_size])

            compare_var = model_name    
    
    heatmaps = np.zeros((len(model_sizes), len(bandlimits)))
    heatmaps_grid = np.zeros((len(model_sizes), len(bandlimits)))
    num_models = len(models2plot)
    clipping_range = (70, 0)
    norm = Normalize(vmin=clipping_range[0], vmax=clipping_range[1], clip=True)
    clipping_range = (50, -50)  
    norm2 = Normalize(vmin=clipping_range[0], vmax=clipping_range[1], clip=True)
    label_font_size = 18
    fig = plt.figure(figsize=(30, 12))
    gs = gridspec.GridSpec(2, 8, height_ratios=[1, 1], hspace=-0.6, wspace=0.02)
    for row_idx in range(2):
        for model_idx, model_name in enumerate(models2plot):
            for j, model_size in enumerate(model_sizes):
                for i, bandlimit in enumerate(bandlimits):
                    if task_name == '2d_StarTarget':
                        heatmaps[j] = psnrs[model_name][j][::-1]
                        heatmaps_grid[j] = np.array(psnrs[model_name][j][::-1]) - np.array(psnrs['Grid'][j][::-1])
                    else:
                        if model_name == 'GSplat' and '3d' in task_name:
                            heatmaps[j][i] = np.nan
                            heatmaps_grid[j][i] = np.nan
                        else:
                            heatmaps[j][i] = psnr_mean[model_name][model_size][i]
                            heatmaps_grid[j][i] = psnr_mean[model_name][model_size][i] - psnr_mean['Grid'][model_size][i]
                
            if row_idx == 0:
                plt.subplot(gs[0, model_idx])
                im1 = plt.imshow(heatmaps, aspect='auto', cmap='viridis', norm=norm)
                if task_name == '2d_SparseSphereSignal':
                    plt.title(f'{plot_model_name(model_name)}', fontproperties = times_new_roman) 
                plt.xticks([])
            elif row_idx == 1:
                plt.subplot(gs[1, model_idx])
                im2 = plt.imshow(heatmaps_grid, aspect='auto', cmap='coolwarm', norm=norm2)
                plt.xticks(ticks=range(len(bandlimits)), labels=bandlimits_vis, fontsize=label_font_size) 
            plt.gca().set_box_aspect(0.90)
            
            if model_idx == 0:
                plt.yticks(ticks=range(len(model_sizes)), labels=model_sizes_exp, fontsize = label_font_size) 
            else:
                plt.yticks([]) 
    cax1 = plt.gcf().add_axes([0.905, 0.52, 0.007, 0.18]) 
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label(label='PSNR (dB)', fontsize=label_font_size, labelpad = 22)
    cbar1.ax.tick_params(labelsize=label_font_size)

    cax2 = plt.gcf().add_axes([0.905, 0.3, 0.007, 0.18]) 
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.set_label(label='PSNR Gap (dB)', fontsize=label_font_size)
    cbar2.ax.tick_params(labelsize=label_font_size)
    fig.text(0.08, 0.5, f'{task_names_vis[task_idx]}', rotation=90, va='center', ha='center', fontproperties=times_new_roman_task)
    if '3d_Bandlimit' in task_name:
        fig.text(0.5, 0.22, 'Bandwidths',  va='center', ha='center', fontproperties=times_new_roman_task)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    
    plt.savefig(f'heatmaps/{task_name}_heatmap_total.pdf', bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    plt.close()
    # break
    with open(f"heatmaps/eval_synthetic_{task_name}_metrics.pkl", "wb") as f:
        if task_name[0] == '3':
            pickle.dump({"psnr": psnr_mean, "psnr_std": psnr_std, "iou": iou_mean, "iou_std": iou_std}, f)
        else:
            pickle.dump({"psnr": psnr_mean, "psnr_std": psnr_std, "ssim": ssim_mean, "ssim_std": ssims_std, "lpips": lpips_mean, "lpips_std": lpipses_std}, f)