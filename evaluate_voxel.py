import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bandlimited_signal import *
from tqdm import tqdm
# from utils import plot_colormap, set_colormap, plot_linestyle
sys.path.append('models/')
from models.inr import *
from utils import *
import pickle
gpu_num = 0
device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(f'cuda:{gpu_num}')

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
        # print(item)
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

signals = [SparseSphereSignal, BandlimitedSignal, Sierpinski, StarTarget, Voxel_Fitting]
task_names = ['3d_VoxelFitting']
model_sizes = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
model_sizes_int = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
seeds = ['1234']

sparse = False
sparses = [False, True]
for sparse in sparses:
    task_bar = tqdm(task_names)
    for task_name in task_bar:
        for i, signal in enumerate(signals):
            if task_name.split("_")[-1] == signal.__name__:
                break
        if sparse:
            search_directory = f"band_limit_figs/{task_name}_sparse"  # Directory path to search in
            save_name = f'{task_name}_sparse'
        else:
            search_directory = f"band_limit_figs/{task_name}"  # Directory path to search in
            save_name = f'{task_name}'
        target_folder_name = "complete"  # Name of the folder to search for
        models = ['Grid', 'Grid_TV', 'FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'BACON']
        models2plot = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'BACON' ,'Grid', 'Grid_TV']
        
        task_bar.set_description(f'{task_name}, {models}')

        colors = set_colormap(len(models))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        num_bandwidth = 9
        if task_name == '2d_StarTarget' or task_name == '3d_VoxelFitting':
            num_bandwidth = 1    
            bandlimits = [0.1]

        num_params = 6

        mean_med = 'median'
        view_all = True
        target_folders = {}
        psnrs = {}
        psnr_std = {}
        psnr_mean = {}
        for model_name in models:
            if model_name == 'Grid_TV':
                model_name_temp = 'Grid'
            else: model_name_temp = model_name

            psnrs[model_name] = {}
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
            if len(df) == 0:
                print(f"Skipping {model_name} due to missing data")
                continue
            dfs = eval_dataFrame(df, model_name)
            print(model_name, target_folders[model_name])
            outputs = {}
            
            if task_name == task_names[-1]:
                seeds = ['1234']
            for model_size in model_sizes:
                outputs[model_size] = {}
                psnrs[model_name][model_size] = []
                psnr_std[model_name][model_size] = []
                psnr_mean[model_name][model_size] = []
                for bandlimit_idx, bandlimit in enumerate(bandlimits):
                    outputs[model_size][str(bandlimit)] = {}
                    psnrs[model_name][model_size].append([])
                    psnr_std[model_name][model_size].append([])
                    psnr_mean[model_name][model_size].append([])
            compare_var = model_name

        super_res_list = [False, True]
        for super_res in super_res_list:
            if super_res == True:
                save_name = f'{save_name}_SR'
            else:
                save_name = save_name
            psnrs ={}
            ious = {}
            target_folders = {}
            dimension = 3
            signal = Voxel_Fitting(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution=super_res, sparse= sparse)
            signal = torch.tensor(signal.signal, dtype=torch.float).to(device)
            signal_orig = Voxel_Fitting(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution=False, sparse= sparse)
            
            signal_shape = signal_orig.signal.shape
            num_samples = len(signal.reshape(-1))
            # fake initialization
            coords_y = np.linspace(-1, 1, signal.shape[0], endpoint=False)
            coords_x = np.linspace(-1, 1, signal.shape[1], endpoint=False)
            coords_z = np.linspace(-1, 1, signal.shape[2], endpoint=False)
            x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y, coords_z), -1), dtype=torch.float).to(device)
            x = x.reshape(-1, dimension)
            loss_fn = nn.MSELoss().to(device)

            batch_size = int(4e5)
            num_batches = (num_samples + batch_size - 1) // batch_size
            signal = signal.reshape(num_samples)
            for model_name in models:
                if model_name == 'BACON':
                    signal = Voxel_Fitting(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution=super_res, sparse= sparse)
                    signal = torch.tensor(signal.signal, dtype=torch.float).to(device)
                    signal_orig = Voxel_Fitting(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution=False, sparse= sparse)
                    
                    signal_shape = signal_orig.signal.shape
                    num_samples = len(signal.reshape(-1))
                    coords_y = np.linspace(-0.5, 0.5, signal.shape[0], endpoint=False)
                    coords_x = np.linspace(-0.5, 0.5, signal.shape[1], endpoint=False)
                    coords_z = np.linspace(-0.5, 0.5, signal.shape[2], endpoint=False)
                    x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y, coords_z), -1), dtype=torch.float).to(device)
                    x = x.reshape(-1, dimension)
                    batch_size = int(4e5)
                    num_batches = (num_samples + batch_size - 1) // batch_size
                    signal = signal.reshape(num_samples)
                results = {}
                psnrs[model_name] = {}
                ious[model_name] = {}
                target_folders[model_name] = []
                found_folders = find_folder(target_folder_name, search_directory)
                df = []

                for folder_name in found_folders:
                    if model_name in folder_name:
                        if model_name == 'FourFeat':
                            if 'FourFeat_Embed' not in folder_name and 'FourFeat_5e3' not in folder_name and 'FourFeat_Triple' not in folder_name:
                                target_folders[model_name].append(folder_name)
                                df.append(pd.read_csv(folder_name + '/records.csv'))
                        elif model_name == 'Grid':
                            if 'Grid_TV' not in folder_name:
                                target_folders[model_name].append(folder_name)
                                df.append(pd.read_csv(folder_name + '/records.csv'))
                        else: 
                            target_folders[model_name].append(folder_name)
                            df.append(pd.read_csv(folder_name + '/records.csv'))
                else:
                    pass
            
                for model_size in model_sizes:
                    if model_name == 'GSplat':
                        continue
                    results[model_size] = {}
                    psnrs[model_name][model_size] = []
                    ious[model_name][model_size] = []
                    for bandlimit_idx, bandlimit in enumerate(bandlimits):
                        results[model_size][str(bandlimit)] = {}
                        psnrs[model_name][model_size].append([])
                        ious[model_name][model_size].append([])
                        for i, seed in enumerate(seeds):
                            results[model_size][str(bandlimit)][target_folders[model_name][i].split("_")[-2]] = f'{target_folders[model_name][i]}/{model_name}_{model_size}_{bandlimit}_{int(target_folders[model_name][i].split("_")[-2])}.pth'
                            
                seeds = [1234]
                
                torch.cuda.empty_cache()
                

                if model_name == 'Grid_TV':
                    model_class = get_model('Grid')
                else:
                    model_class = get_model(model_name)
                for seed_idx, seed in enumerate(seeds):
                    torch.cuda.empty_cache()
                    plt.figure(2, figsize = (7,12))
                    for i, (model_size, max_params) in enumerate(zip(model_sizes, model_sizes_int)):
                        if model_name == 'GSplat':
                            if max_params > 1e6:
                                continue
                        # if super_res:
                        torch.cuda.empty_cache()
                        output_whole = torch.zeros_like(signal)
                        set_seed(seed)
                        signal = Voxel_Fitting(dimension, length = 1000, bandlimit = 0.1, seed = 1234, super_resolution=super_res, sparse = sparse)
                        signal = torch.tensor(signal.signal, dtype=torch.float).to(device)
                        trn_dataset, dataloader = init_dataloader(signal, 1, signal_shape)
                        recon_signal_shape = signal.shape
                        signal = signal.reshape(num_samples)
                        if model_name in  ['GAPlane', 'GSplat', 'BACON']:
                            inr = model_class(dimension, max_params, resolution = signal_shape).to(device)
                        else:
                            inr = model_class(dimension, max_params).to(device)
                        inr.load_state_dict(torch.load(results[model_size][str(bandlimit)]['1234'])['model_state_dict'])
                        with torch.no_grad():
                            inr.eval()
                            if model_name == 'GSplat':
                                output_whole = inr(x).squeeze()
                            else:
                                if model_name == 'BACON':
                                    train_generator = iter(dataloader)
                                for _, batch_idx in enumerate(range(num_batches)):
                                    start_idx = batch_idx * batch_size
                                    end_idx = min(start_idx + batch_size, num_samples)
                                    x_batch = x[start_idx:end_idx].to(device)
                                    signal_batch = signal[start_idx:end_idx].to(device)
                                    if model_name == 'BACON':
                                        output_whole[start_idx:end_idx] = inr(x_batch)[-1].squeeze()
                                    else: 
                                        output_whole[start_idx:end_idx] = inr(x_batch).squeeze()
                                    output_whole = torch.clamp(output_whole, min=0.0, max=1.0)
                        
                        loss_val = loss_fn(output_whole, signal).cpu().squeeze()
                        psnr = -10*torch.log10(loss_val).detach().numpy()
                        iou_val = compute_iou(output_whole, signal, device)
                        psnrs[model_name][model_size][0].append(psnr)
                        ious[model_name][model_size][0].append(iou_val)
                        print_memory_usage(device, tag=f"{model_name} {seed} {model_size} PSNR {psnr:2f} iou {iou_val:4f}")                
                        del inr
                        output_whole = output_whole.reshape(recon_signal_shape)
                        output_whole = output_whole.squeeze().cpu().detach().numpy()
                        plt.figure(1)
                        plt.imshow(output_whole[:,:,len(output_whole[0,0])//2])
                        plt.title(f'{model_size} PSNR {psnr:.2f}')
                        if super_res:
                            np.save(f'{search_directory}/{model_name}_{model_size}_sr.npy',output_whole)
                            plt.savefig(f'{search_directory}/{model_name}_{model_size}_sr.png')
                            plt.savefig(f'{search_directory}/{model_name}_{model_size}_sr.jpeg')
                        else:
                            np.save(f'{search_directory}/{model_name}_{model_size}.npy',output_whole)
                            plt.savefig(f'{search_directory}/{model_name}_{model_size}.png')
                            plt.savefig(f'{search_directory}/{model_name}_{model_size}.jpeg')
                        plt.close(1)
                        plt.figure(2)
                        plt.subplot(3,2,i+1)
                        plt.imshow(output_whole[:,:,len(output_whole[0,0])//2])
                        plt.title(f'{model_size} PSNR {psnr:.2f}')
                    plt.tight_layout()
                    if super_res:   
                        plt.savefig(f'{search_directory}/{model_name}_sr.png')
                        plt.savefig(f'{search_directory}/{model_name}_sr.jpeg')
                    else:
                        plt.savefig(f'{search_directory}/{model_name}.png')
                        plt.savefig(f'{search_directory}/{model_name}.jpeg')
                    plt.close(2)
            
            plt.figure()
            psnrs2save =[]
            for i, model_name in enumerate(models2plot):
                model_sizes = list(psnrs[model_name].keys())  
                psnr_values = [psnrs[model_name][model_size][0] for model_size in model_sizes]
                psnrs2save.append(psnr_values)
                if super_res:
                    if model_name == 'Grid_TV':
                        model_name = 'Grid'
                    elif model_name == 'Grid':
                        continue
                else:
                    if model_name == 'Grid':
                        pass
                    elif model_name == 'Grid_TV':
                        continue
                plt.plot(psnr_values, label=f'{model_name}',  c = plot_colormap(model_name), linestyle = plot_linestyle(model_name), marker = plot_marker(model_name), markevery=2)
                plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
                
            if super_res == True:
                plt.legend(loc='lower left')
                plt.savefig(f'{search_directory}/voxel_results_sr.png')
                plt.savefig(f'{search_directory}/voxel_results_sr.jpeg')
                np.save(f'{search_directory}/voxel_results_sr.npy', np.array(psnrs2save, dtype=object))
            else:
                plt.legend(loc='upper left')
                plt.savefig(f'{search_directory}/voxel_results.png')
                plt.savefig(f'{search_directory}/voxel_results.jpeg')
                
                np.save(f'{search_directory}/voxel_results.npy', np.array(psnrs2save, dtype=object))
            plt.close()
            
            plt.figure()
            psnrs2save_grid = []
            for i, model_name in enumerate(models):
                model_sizes = list(psnrs[model_name].keys()) 
                psnr_values = np.array([psnrs[model_name][model_size][0] for model_size in model_sizes])
                
                if super_res:
                    if model_name == 'Grid_TV':
                        grid_sizes = model_sizes
                        grid_psnrs = np.array(psnr_values)
                        continue
                    if model_name == 'Grid':
                        continue
                else:
                    if model_name == 'Grid':
                        grid_sizes = model_sizes
                        grid_psnrs = np.array(psnr_values)
                        continue
                    if model_name == 'Grid_TV':
                        continue
                try:
                    diff = psnr_values - grid_psnrs
                    psnrs2save_grid.append(diff)
                    plt.plot(diff, label=f'{model_name}',
                            c=plot_colormap(model_name),
                            linestyle=plot_linestyle(model_name),
                            marker=plot_marker(model_name), markevery=2)
                except ValueError as e:
                    print(f"[Skip] {model_name}: PSNR shape mismatch – {e}")
            plt.axhline(y=0, color='gray', linestyle='--', alpha = 0.2)
            plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes)
            
            if super_res == True:
                plt.legend(loc='lower left')
                plt.savefig(f'{search_directory}/voxel_results_grid_sr.png')
                plt.savefig(f'{search_directory}/voxel_results_grid_sr.jpeg')
                np.save(f'{search_directory}/voxel_results_grid_sr.npy', psnrs2save_grid)
            else:
                plt.legend(loc='upper left')
                plt.savefig(f'{search_directory}/voxel_results_grid.png')
                plt.savefig(f'{search_directory}/voxel_results_grid.jpeg')
                np.save(f'{search_directory}/voxel_results_grid.npy', psnrs2save_grid)
            plt.close()
            with open(f"heatmaps/eval_{save_name}_metrics.pkl", "wb") as f:
                pickle.dump({"psnr": psnrs, "iou": ious}, f)