import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from bandlimited_signal import *
from utils import *
import sys
sys.path.append('models/')
from models.inr import *
from utils import *
import matplotlib.patches as patches
from matplotlib.transforms import Bbox
import lpips
from skimage.metrics import structural_similarity as ssim
import json
import pickle

gpu_num = 0
device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')


lpips_fn = lpips.LPIPS(net='vgg').to(device)
     
os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
ratio = 0.35
small_window_ratio = 0.35
region_ratio = [0.450980, 0.32448378]
import json
torch.set_default_dtype(torch.float)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
colors = [cm.tab20(i) for i in range(20)]
vis_seed = False
cutting_factor = 4

torch.set_default_device(f'cuda:{gpu_num}')
imgs = list(range(0,10))

dimension = 2
model_sizes = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
model_sizes_int = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]

tasks = ['super_resolution', 'denoising', 'overfitting']
root_dir = 'band_limit_figs/1RealImage'
noises = [0.05, 0.1]
for task in tasks:
    search_directory = f"{root_dir}/{tasks[0]}/"
    for noise in noises:
        if task == tasks[0]:
            is_super_resolution = True
            search_directory = f"{root_dir}/{task}/"
            if noise == 0.1:
                continue
        elif task == tasks[1]:
            is_super_resolution = False
            search_directory = f'{root_dir}/{task}/{noise}/'
        elif task == tasks[2]:
            is_super_resolution = False
            search_directory = f"{root_dir}/{tasks[0]}/"
            if noise == 0.1:
                continue
        psnrs_imgs = []
        ssims_imgs = []
        lpipses_imgs = []
        for img_idx in imgs:
            print(f'img index is {img_idx}')
            signal = RealImage(dimension = 2, length = 1000, bandlimit=0.1, seed = img_idx, super_resolution=is_super_resolution)
            target = np.array(signal.signal)
            print(f'target shape : {target.shape}')
            
            coords_y = np.linspace(-1, 1, target.shape[0], endpoint=False)
            coords_x = np.linspace(-1, 1, target.shape[1], endpoint=False)
            coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
            x = torch.tensor(coords, dtype=torch.float).to(device)
            models_list = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'GSplat', 'BACON', 'Grid', 'Grid_TV']
            
            # print(models_list)
            results = {}
            for model_name in models_list:
                if model_name not in results:
                    results[model_name] = {"default": {}, "best": {}}  
                if model_name == 'BACON':
                    coords_y = np.linspace(-0.5, 0.5, target.shape[0], endpoint=False)
                    coords_x = np.linspace(-0.5, 0.5, target.shape[1], endpoint=False)
                    coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
                    x = torch.tensor(coords, dtype=torch.float).to(device)
                else:
                    coords_y = np.linspace(-1, 1, target.shape[0], endpoint=False)
                    coords_x = np.linspace(-1, 1, target.shape[1], endpoint=False)
                    coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
                    x = torch.tensor(coords, dtype=torch.float).to(device)
                for dirs, _, files in os.walk(search_directory):
                    # print(dirs, files)
                    if f'img_{img_idx}' in dirs:
                        if len(files) != 0:
                            for file in files:
                                if ".pth" in file:
                                    if "best" not in file:  
                                        if model_name in file:
                                            current_model_name = model_name
                                        elif 'Grid' in file and 'Grid_TV' not in file:
                                            current_model_name = 'Grid'
                                        else:
                                            continue
                                        if current_model_name not in results:
                                            results[current_model_name] = {"default": {}, "best": {}}

                                        seed_val, params_val = file[-8:-4], file[-18:-13]
                                        if params_val in model_sizes:
                                            if params_val not in results[current_model_name]["default"]:
                                                results[current_model_name]["default"][params_val] = {}
                                            results[current_model_name]["default"][params_val][seed_val] = os.path.join(dirs, file)

                                    elif "best" in file:  
                                        if model_name in file:
                                            current_model_name = model_name
                                        elif 'Grid' in file and 'Grid_TV' not in file:
                                            current_model_name = 'Grid'
                                        else:
                                            continue
                                        if current_model_name not in results:
                                            results[current_model_name] = {"default": {}, "best": {}}

                                        parts = file.split('_')
                                        params_val = parts[-3]  # e.g., '3e+06'
                                        seed_val = parts[-1].split('.')[0]  # e.g., '2024'

                                        if params_val in model_sizes:
                                            if params_val not in results[current_model_name]["best"]:
                                                results[current_model_name]["best"][params_val] = {}
                                            results[current_model_name]["best"][params_val][seed_val] = os.path.join(dirs, file)
            for model_name in models_list:
                for params_val in model_sizes:
                    
                    print(model_name, params_val, results[model_name]['default'][params_val]['1234'])
            H,W,_ = target.shape
            Hy, Wx = H, W
            print(f'Img index {img_idx} H {Hy} W {Wx}')
            if is_super_resolution:
                H, W = H//cutting_factor, W//cutting_factor
            batch_shape = (H,W,3)
            temp_img = np.zeros_like(signal.signal)
            seeds = [1234]
            models_psnrs_default, models_psnrs_best = [], []
            models_std_default, models_std_best = [], []
            models_ssims_best, models_lpipses_best = [], []
            models_ssim_std_default, models_ssim_std_best = [], []
            models_lpips_std_default, models_lpips_std_best = [], []
            torch.cuda.empty_cache()
            ssims = {}
            lpipses = {}
            psnrs = {}
            for model_idx, model_name in enumerate(models_list):
                if model_name == 'BACON':
                    coords_y = np.linspace(-0.5, 0.5, target.shape[0], endpoint=False)
                    coords_x = np.linspace(-0.5, 0.5, target.shape[1], endpoint=False)
                    coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
                    x = torch.tensor(coords, dtype=torch.float).to(device)
                else:
                    coords_y = np.linspace(-1, 1, target.shape[0], endpoint=False)
                    coords_x = np.linspace(-1, 1, target.shape[1], endpoint=False)
                    coords = np.stack(np.meshgrid(coords_x, coords_y), -1)
                    x = torch.tensor(coords, dtype=torch.float).to(device)
                psnrs_default, psnrs_best = {}, {}
                ssims_default, ssims_best = {}, {}
                lpipses_default, lpipses_best = {}, {}
                models_default, models_best = [], []
                psnrs[model_name] = {}
                ssims[model_name] = {}
                lpipses[model_name] = {}
                generalized_imgs_default = np.zeros((len(models_list), len(seeds), len(model_sizes), target.shape[0], target.shape[1], target.shape[2]))
                generalized_imgs_best = np.zeros_like(generalized_imgs_default)
                if model_name == 'Grid_TV':
                    model_class = get_model('Grid')
                else:
                    model_class = get_model(model_name)
                for seed_idx, seed in enumerate(seeds):
                    ssims[model_name][seed] = {}
                    lpipses[model_name][seed] = {}
                    psnrs[model_name][seed] = {}
                    if vis_seed:
                        plt.figure(figsize=(7,8))
                    outputs_default, outputs_best = [], []
                    torch.cuda.empty_cache()

                    psnrs_default[str(seed)] = []
                    psnrs_best[str(seed)] = []
                    ssims_default[str(seed)] = []
                    ssims_best[str(seed)] = []
                    lpipses_default[str(seed)] = []
                    lpipses_best[str(seed)] = []
                    for i, (max_params_str, max_params) in enumerate(zip(model_sizes, model_sizes_int)):
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        print_memory_usage(device, tag=f"{model_name} {seed} {max_params_str}")
                        default_path = results[model_name]["default"][max_params_str][str(seed)]
                        print(default_path)
                        if model_name == 'GAPlane':
                            inr_default = model_class(dimension, max_params, out_features=3, resolution = [Wx,Hy]).to(device)
                        elif model_name == "GSplat" or model_name == 'BACON':
                            inr_default = model_class(dimension, max_params, out_features = 3, resolution = [Hy, Wx]).to(device)
                        else:    
                            inr_default = model_class(dimension, max_params, out_features=3).to(device)
                        inr_default.load_state_dict(torch.load(default_path)['model_state_dict'])
                        models_default.append(inr_default)
                        if not is_super_resolution:# or model_name == 'Grid':
                            if model_name == 'GSplat':
                                temp_img = inr_default(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                            elif model_name =='BACON':
                                temp_img = inr_default(x)[-1].reshape(target.shape).squeeze().cpu().detach().numpy()
                            else:
                                temp_img = inr_default(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                            
                        else:
                            if model_name == 'GSplat':
                                temp_img = inr_default(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                            else:
                                for xi in range(cutting_factor):
                                    for yi in range(cutting_factor):
                                        x_batch = x[H * (xi):H * (xi + 1), W * (yi):W * (yi + 1), :]
                                        if model_name =='BACON':
                                            temp = inr_default(x_batch)[-1].squeeze().cpu().detach().numpy()
                                        else:
                                            temp = inr_default(x_batch).squeeze().cpu().detach().numpy()
                                        temp = temp.reshape(batch_shape)
                                        temp_img[H * xi:H * (xi + 1), W * yi:W * (yi + 1), :] = temp
                        temp_img = np.clip(temp_img, 0, 1)
                        outputs_default.append(temp_img)

                        generalized_imgs_default[model_idx, seed_idx, i] = temp_img
                        psnr = PSNR(outputs_default[i], signal)
                        # print(signal.signal.shape)
                        ssim_val = ssim(outputs_default[i], signal.signal, channel_axis=-1, data_range=signal.signal.max() - signal.signal.min())
                        lpips_val = compute_lpips(outputs_default[i], signal.signal, device)
                        psnrs[model_name][seed][max_params] = {'default' : psnr}
                        ssims[model_name][seed][max_params] = {'default' : ssim_val}
                        lpipses[model_name][seed][max_params] = {'default' : lpips_val}
                        if vis_seed:
                            plt.subplot(3,2,i+1)
                            plt.imshow(outputs_default[i])
                            plt.title(f'PSNR {psnr:.2f}')
                            plt.colorbar()
                        psnrs_default[str(seed)].append(psnr)
                        ssims_default[str(seed)].append(ssim_val)
                        lpipses_default[str(seed)].append(lpips_val)

                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        best_path = results[model_name]["best"][max_params_str][str(seed)]
                        if model_name == 'GAPlane':
                            inr_best = model_class(dimension, max_params, out_features=3, resolution = [Wx,Hy]).to(device)
                        elif model_name =='GSplat' or model_name == 'BACON':
                            inr_best = model_class(dimension, max_params, out_features = 3, resolution = [Hy, Wx]).to(device)
                        else:    
                            inr_best = model_class(dimension, max_params, out_features=3).to(device)
                        inr_best.load_state_dict(torch.load(best_path)['model_state_dict'])
                        models_best.append(inr_best)
                        if not is_super_resolution:
                            if model_name == 'GSplat':
                                temp_img = inr_best(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                            if model_name =='BACON':
                                temp_img = inr_best(x)[-1].squeeze().cpu().detach().numpy()
                            else:
                                temp_img = inr_best(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                        else:
                            if model_name == 'GSplat':
                                temp_img = inr_best(x).reshape(target.shape).squeeze().cpu().detach().numpy()
                            else:
                                for xi in range(cutting_factor):
                                    for yi in range(cutting_factor):
                                        x_batch = x[H * (xi):H * (xi + 1), W * (yi):W * (yi + 1), :]
                                        if model_name =='BACON':
                                            temp = inr_best(x_batch)[-1].squeeze().cpu().detach().numpy()
                                        else:
                                            temp = inr_best(x_batch).squeeze().cpu().detach().numpy()
                                        temp = temp.reshape(batch_shape)
                                        temp_img[H * xi:H * (xi + 1), W * yi:W * (yi + 1), :] = temp
                        temp_img = np.clip(temp_img, 0, 1)
                        outputs_best.append(temp_img)

                        generalized_imgs_best[model_idx, seed_idx, i] = temp_img
                        psnr = PSNR(outputs_best[i], signal)
                        ssim_val = ssim(outputs_best[i], signal.signal, channel_axis=-1, data_range=signal.signal.max() - signal.signal.min())
                        lpips_val = compute_lpips(outputs_best[i], signal.signal, device)
                        psnrs[model_name][seed][max_params] = {'best' : psnr}
                        ssims[model_name][seed][max_params] = {'best' : ssim_val}
                        lpipses[model_name][seed][max_params] = {'best' : lpips_val}
                        psnrs_best[str(seed)].append(psnr)
                        ssims_best[str(seed)].append(ssim_val)
                        lpipses_best[str(seed)].append(lpips_val)
                    # inferred_imgs[str(seed)] = outputs
                    if vis_seed:
                        plt.tight_layout()
                        plt.show()
                    
                if task == 'denoising':
                    task_name = f'{task}_{noise}'
                else: task_name = f'{task}'
                
                save_root = f'{root_dir}/{img_idx}'
                avg_imgs_default = np.mean(generalized_imgs_default, axis=1)
                avg_imgs_best = np.mean(generalized_imgs_best, axis=1)
                mean_psnr_default = np.mean(list(psnrs_default.values()), axis=0)
                std_psnr_default = np.std(list(psnrs_default.values()), axis=0)
                mean_psnr_best = np.mean(list(psnrs_best.values()), axis=0)
                mean_ssim_best = np.mean(list(ssims_best.values()), axis=0)
                mean_lpips_best = np.mean(list(lpipses_best.values()), axis=0)
                std_psnr_best = np.std(list(psnrs_best.values()), axis=0)
                std_ssims_default = np.std(list(ssims_default.values()), axis=0)
                std_ssims_best = np.std(list(ssims_best.values()), axis=0)
                std_lpipses_default = np.std(list(lpipses_default.values()), axis=0)
                std_lpipses_best = np.std(list(lpipses_best.values()), axis=0)
                models_psnrs_default.append(mean_psnr_default)
                models_psnrs_best.append(mean_psnr_best)
                models_ssims_best.append(mean_ssim_best)
                models_lpipses_best.append(mean_lpips_best)
                models_std_default.append(std_psnr_default)
                models_std_best.append(std_psnr_best)
                models_ssim_std_default.append(std_ssims_default)
                models_ssim_std_best.append(std_ssims_best)
                models_lpips_std_default.append(std_lpipses_default)
                models_lpips_std_best.append(std_lpipses_best)
                plt.figure()
                plt.plot(mean_psnr_default, label=f'{model_name} Default', linestyle='--', c = colors[i])
                plt.plot(mean_psnr_best, label=f'{model_name} Best', linestyle='-')
                plt.title(f'{model_name} PSNR Comparison')
                plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
                plt.legend()
                plt.savefig(f'{save_root}/{model_name}_{task_name}_psnr_comparison.png')
                plt.savefig(f'{save_root}/{model_name}_{task_name}_psnr_comparison.jpeg')
                np.save(f'{save_root}/{model_name}_{task_name}_psnr_comparison.npy', mean_psnr_best)
                plt.close()

                # Plot images for default

                plt.figure(figsize=(7, 8))
                for i in range(len(model_sizes)):
                    plt.subplot(3, 2, i + 1)
                    plt.imshow(avg_imgs_default[model_idx, i])
                    plt.title(f'{model_sizes[i]} PSNR {PSNR(avg_imgs_default[model_idx, i], signal):.2f}')
                plt.tight_layout()
                plt.savefig(f'{save_root}/{model_name}_{task_name}_images_default.png')
                plt.savefig(f'{save_root}/{model_name}_{task_name}_images_default.jpeg')
                plt.close()
                
                # Plot images for best
                if not os.path.exists(f'{save_root}/../outputs'):
                    os.mkdir(f'{save_root}/../outputs')
                if not os.path.exists(f'{save_root}/../outputs/{img_idx}'):
                    os.mkdir(f'{save_root}/../outputs/{img_idx}')
                plt.figure(figsize=(7, 8))
                for i, model_size in enumerate(model_sizes):
                    plt.subplot(3, 2, i + 1)
                    plt.imshow(avg_imgs_best[model_idx, i])
                    cv2.imwrite(f'{save_root}/../outputs/{img_idx}/{model_name}_{task_name}_{model_size}.png', cv2.cvtColor((avg_imgs_best[model_idx, i]*255).clip(0,255).astype('uint8'), cv2.COLOR_RGB2BGR))
                    np.save(f'{save_root}/../outputs/{img_idx}/{model_name}_{task_name}_{model_size}.npy', avg_imgs_best[model_idx, i])
                    plt.title(f'{model_sizes[i]} PSNR {PSNR(avg_imgs_best[model_idx, i], signal):.2f}')
                plt.tight_layout()
                plt.savefig(f'{save_root}/{model_name}_{task_name}_images_best.png')
                plt.savefig(f'{save_root}/{model_name}_{task_name}_images_best.jpeg')
                plt.close()
            plt.figure()
            for i, model in enumerate(models_list):
                plt.plot(models_psnrs_default[i], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery = 2)
                plt.fill_between(range(len(model_sizes)), models_psnrs_default[i]+ models_std_default[i], models_psnrs_default[i] - models_std_default[i], color = colors[i], alpha =0.1)    
            plt.title(f'{task_name} psnrs')
            plt.legend(loc = 'lower right')
            if task == 'overfitting':
                plt.ylim([0,55])
            else: 
                plt.ylim([7.5,32])
            plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
            plt.savefig(f'{save_root}/{task_name}_plots_default.png')
            plt.savefig(f'{save_root}/{task_name}_plots_default.jpeg')
            plt.close()

            plt.figure()
            for i, model in enumerate(models_list):
                plt.plot(models_psnrs_best[i], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery = 2)
                plt.fill_between(range(len(model_sizes)), models_psnrs_best[i]+ models_std_best[i], models_psnrs_best[i] - models_std_best[i], color = colors[i], alpha = 0.1)    
            plt.title(f'{task_name} psnrs best')
            plt.legend(loc = 'lower right')
            if task == 'overfitting':
                plt.ylim([0,55])
            else: 
                plt.ylim([7.5,32])
            plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
            plt.savefig(f'{save_root}/{task_name}_plot_best.png')
            plt.savefig(f'{save_root}/{task_name}_plot_best.jpeg')
            plt.close()

            plt.figure()
            for i, model in enumerate(models_list):
                plt.plot(models_psnrs_best[i], label =plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery = 2)
                plt.plot(models_psnrs_default[i], c = plot_colormap(model_name), linestyle = '--')
                # plt.fill_between(range(len(model_sizes)), models_psnrs_best[i]+ models_std_best[i], models_psnrs_best[i] - models_std_best[i], c = colors[i], alpha = 0.1)    
            plt.title(f'{task_name} psnrs')
            plt.legend(loc = 'lower right')
            if task == 'overfitting':
                plt.ylim([0,55])
            else: 
                plt.ylim([7.5,32])
            plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
            plt.savefig(f'{save_root}/{task_name}_plot_total.png')
            plt.savefig(f'{save_root}/{task_name}_plot_total.jpeg')
            plt.close()
            psnrs_imgs.append(models_psnrs_best)
            ssims_imgs.append(models_ssims_best)
            lpipses_imgs.append(models_lpipses_best)
        
        psnrs_imgs_mean = np.mean(np.array(psnrs_imgs), axis = 0)
        psnrs_imgs_std = np.std(np.array(psnrs_imgs), axis = 0)
        ssims_imgs_mean = np.mean(np.array(ssims_imgs), axis = 0)
        ssims_imgs_std = np.std(np.array(ssims_imgs), axis = 0)
        lpipses_imgs_mean = np.mean(np.array(lpipses_imgs), axis = 0)
        lpipses_imgs_std = np.std(np.array(lpipses_imgs), axis = 0)
        results = {
            "psnr":psnrs_imgs_mean,
            "psnr_std": psnrs_imgs_std,
            "ssim":ssims_imgs_mean,
            "ssim_std":ssims_imgs_std,
            "lpips":lpipses_imgs_mean,
            "lpips_std":lpipses_imgs_std,
        }

        with open(f"{root_dir}/{task_name}_metrics.pkl", "wb") as f:
            pickle.dump(results, f)
        np.save(f'{root_dir}/{task_name}_psnrs_imgs_mean.npy', psnrs_imgs_mean)
        np.save(f'{root_dir}/{task_name}_psnrs_imgs_std.npy', psnrs_imgs_std)
        plt.figure()
        for i, model_name in enumerate(models_list):
            if task == 'overfitting':
                if model_name == 'Grid_TV':
                    continue
            else:
                if model_name == 'Grid':
                    continue
                if model_name == 'Grid_TV':
                    model_name = 'Grid'
            plt.plot(psnrs_imgs_mean[i], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery = 2)
            plt.fill_between(range(len(model_sizes)), psnrs_imgs_mean[i]+ psnrs_imgs_std[i], psnrs_imgs_mean[i] - psnrs_imgs_std[i], color = plot_colormap(model_name), alpha = 0.1)    
        plt.title(f'{task_name} psnrs')
        plt.legend(loc = 'lower right')
        if task == 'overfitting':
            plt.ylim([0,55])
        elif task == 'super_resolution':
            plt.ylim([2.5,27])
        else: 
            plt.ylim([7.5,32])
        plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
        plt.savefig(f'{root_dir}/{task_name}_plot_total_10imgs.png')
        plt.savefig(f'{root_dir}/{task_name}_plot_total_10imgs.jpeg')
        plt.close()
        plt.figure()
        for i, model in enumerate(models_list):
            if task == 'overfitting':
                if model == 'Grid':
                    grid_idx = i
                elif model == 'Grid_TV':
                    continue_idx = i
            else:
                if model == 'Grid_TV':
                    grid_idx = i
                elif model == 'Grid':
                    continue_idx = i
        for i, model_name in enumerate(models_list):
            if i == grid_idx:
                continue
            elif i == continue_idx:
                continue
            else:
                plt.plot(psnrs_imgs_mean[i] - psnrs_imgs_mean[grid_idx], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery = 2)
                plt.fill_between(range(len(model_sizes)), psnrs_imgs_mean[i] - psnrs_imgs_mean[grid_idx] + psnrs_imgs_std[i], psnrs_imgs_mean[i]  - psnrs_imgs_mean[grid_idx] - psnrs_imgs_std[i], color = plot_colormap(model_name), alpha = 0.1)    
        plt.axhline(y=0, color='gray', linestyle='--', alpha = 0.2)
        plt.title(f'{task_name} psnrs')
        
        if task == 'overfitting':
            plt.ylim([np.min(psnrs_imgs_mean[i] - psnrs_imgs_mean[grid_idx])-3,3])
            plt.legend(loc = 'lower left')
        else: 
            plt.legend(loc = 'upper left')
            plt.ylim([-20, 23])
        plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
        plt.savefig(f'{root_dir}/{task_name}_plot_total_10imgs_grid.png')
        plt.savefig(f'{root_dir}/{task_name}_plot_total_10imgs_grid.jpeg')
        plt.close()
