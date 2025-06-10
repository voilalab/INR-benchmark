# 3.79s/it

import numpy as np
import torch
from bandlimited_signal import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import os
import csv
import datetime
import pytz
import sys
import json
import time

seed_list = [1234, 2024, 5678, 7890, 7618] 
img_list = list(range(0,10))
print(img_list)
if len(sys.argv) > 1:
    var_seed, var_cuda, var_task, var_noise = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    print(var_seed, var_cuda, var_task, var_noise)
    selected_seed = seed_list[int(var_seed)]
    cuda_num = int(var_cuda)
    task_num = int(var_task)
    noise_num = int(var_noise)
    tqdm.write(f'{selected_seed}')
else:
    selected_seed = seed_list[0]
    cuda_num = 2
    task_num = 0
    noise_num = 1

dimension = 2

tasks = ['super_resolution', 'denoising', 'overfitting']
task = tasks[task_num]
if task == 'super_resolution':
    noise_level = 0
    is_super_resolution = True
    is_super_resolution = False
    save_folder = task
elif task == 'denoising':
    noise_levels = [0.05, 0.1]
    noise_level = noise_levels[noise_num]
    is_super_resolution = False
    save_folder = os.path.join(task, str(noise_level))
    task = task + f'_{noise_level}'
    
else:
    noise_level = 0
    is_super_resolution = False
set_seed(selected_seed)

sys.path.append('models/')
from utils import *
from models.inr import *

print(cuda_num)

torch.set_default_dtype(torch.float)
torch.set_default_device(cuda_num)
torch.cuda.set_device(cuda_num)

eval_step_size = 100

with open('hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)

models = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'GSplat', 'BACON', 'Grid']

TV_weight = 0
TParams = {
    'signal_class': RealImage,
    'dimension': dimension,
    'iters': 1000,
    'grid_interp': 'bicubic',
}

bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

if TParams['signal_class'].__name__ == 'StarTarget' or TParams['signal_class'].__name__ == 'RealImage': 
    bandlimits = [0.1]

now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")

batch_training = False
dimension = TParams['dimension']

if dimension == 2:
    visualize = True
    model_sizes = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
elif dimension == 3:
    model_sizes = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
    visualize = False
    TParams['grid_interp'] = 'bilinear'

signal_class = TParams['signal_class']
signal_name = signal_class.__name__
if signal_name == 'RealImage':
    color_channel = 3
else:
    color_channel = 1
grid_class = get_model('Grid')

num_steps = TParams['iters']
interpolation_of_grid = TParams['grid_interp']


for img_idx in img_list:
    signal = TParams['signal_class'](dimension, length = 1000, bandlimit = 0.1, seed = img_idx, super_resolution = is_super_resolution)
    signal = torch.tensor(signal.signal, dtype=torch.float)
    Hy, Wx, _ = signal.shape
    signal_shape = signal.shape
    coords_y = np.linspace(-1, 1, signal.shape[0], endpoint=False)
    coords_x = np.linspace(-1, 1, signal.shape[1], endpoint=False)
    x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y), -1), dtype=torch.float).cuda()
    
    num_samples = signal.shape[0]*signal.shape[1]
    signal_shape = signal.shape
    loss_fn_bacon = multiscale_image_mse
    loss_fn = nn.MSELoss().cuda()

    if batch_training:
        x = x.reshape(-1, dimension)

    model_bar = tqdm(models)
    for model in model_bar:
        if model == 'BACON':
            coords_y = np.linspace(-0.5, 0.5, signal.shape[0], endpoint=False)
            coords_x = np.linspace(-0.5, 0.5, signal.shape[1], endpoint=False)
            x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y), -1), dtype=torch.float).cuda()
            if batch_training:
                x = x.reshape(-1, dimension)
            trn_dataset, dataloader = init_dataloader(signal, res = [Hy, Wx])
        TV_weight = 0
        if model == 'Grid_TV':
            model = 'Grid'
            TV_weight = [1e-2, 1e-2, 5e-2, 1e-1, 1e-1, 1e0]
        model_bar.set_description(f'{model}')
        model_class = get_model(model)
        model_name = model_class.__name__
        if TV_weight != 0:
            model_name = model+'_TV'

        learning_rate = hyperparams['models'][model]['lr']
                
        notes = f'{dimension}d_{signal_name}_{model_name}_lr_{learning_rate:0.0e}_{num_steps}_{task}_{selected_seed}'
        tqdm.write(notes)
        save_name = f'band_limit_figs/1RealImage/{save_folder}/{formatted_time}_{notes}_img_{img_idx}/'
        folder4eval = f'band_limit_figs/1RealImage/{img_idx}'
        make_folders(folder4eval)
        make_folders(f'band_limit_figs/1RealImage/overfitting')
        target_dir = make_folders(save_name)
        # If the CSV file doesn't exist, create it and write the header
        csv_filename = save_name + 'records.csv'
        iters_psnrs_name = save_name + 'iters_psnrs.csv'
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as f:
                fieldnames = ['bandlimits', 'num_params', 'sigma', 'omega', 'grid', model_name]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        if not os.path.exists(iters_psnrs_name):
            with open(iters_psnrs_name, 'w', newline='') as f:
                fieldnames_iters = ['bandlimits', 'num_params', 'iters', 'psnrs']
                writer_iter = csv.DictWriter(f, fieldnames=fieldnames_iters)
                writer_iter.writeheader()
        max_params_bar = tqdm(model_sizes)
        for model_size_idx, max_params in enumerate(max_params_bar):
            max_params_bar.set_description(f"{task} img {img_idx} {formatted_time} cuda {cuda_num} {selected_seed}")
            
            bandlim_bar = tqdm(bandlimits)

            if batch_training:
                # batch size initialization
                if model_name == 'GSplat':
                    batch_size = num_samples
                elif model_name == 'Wire':
                    if max_params in [1e6, 3e6]:
                        batch_size = 4e5
                    else: batch_size = num_samples
                else:
                    if max_params in  [3e6]:
                        batch_size = 4e5 
                    else: batch_size = num_samples
                batch_size = int(batch_size)
                num_batches = (num_samples + batch_size - 1) // batch_size  # Number of mini-batches
            
            for bandlimit in bandlim_bar:
                set_seed(selected_seed)
                best_psnr = 0  
                best_model_state = None
                
                signal = TParams['signal_class'](dimension, length = 1000, bandlimit = bandlimit, seed = img_idx)
                signal = torch.tensor(signal.signal, dtype=torch.float).cuda()
                signal = signal + torch.randn_like(signal)*noise_level
                if batch_training:
                    signal = signal.reshape(num_samples, color_channel).squeeze()
                
                if model in ['GAPlane', 'GSplat', 'BACON']:
                    inr = model_class(dimension, max_params, out_features = 3, resolution = signal_shape).cuda()
                else:
                    inr = model_class(dimension, max_params, out_features = 3).cuda()
                if 'FourFeat' in model:
                    np.save(f'{save_name}/{model}_BMat_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',inr.B.cpu().numpy())
                optim = torch.optim.Adam(lr=learning_rate, params=inr.parameters())
                
                model_psnrs = []
                model_iters = []

                bandlim_bar.set_description(f"{dimension}D {signal_name} band limit {bandlimit}, lr={learning_rate:0.0e}")

                grid = grid_class(dimension=dimension, max_params=max_params, out_features = 3,interpolation=interpolation_of_grid).cuda()
                optim_grid = torch.optim.Adam(lr=1e-1, params=[grid.grid])
                for iter in tqdm(range(num_steps), desc='grid', leave=False):
                    output = grid(x)
                    if not batch_training:
                        output = output.reshape(signal_shape)
                    loss = loss_fn(output, signal)
                    optim_grid.zero_grad()
                    loss.backward()
                    optim_grid.step()  
                    if iter == num_steps -1:
                        with torch.no_grad():
                            output = grid(x)
                            if not batch_training:
                                output = output.reshape(signal_shape)
                            loss = loss_fn(output, signal)
                            loss_val = loss_fn(output, signal).cpu().squeeze()
                            psnr = -10*torch.log10(loss_val).numpy()
                            grid_psnr = psnr
                
                torch.cuda.empty_cache()
                if model_name == 'Grid':
                    output2save = output.reshape(signal_shape).squeeze().detach().cpu().numpy()
                    plt.figure()
                    output2save = np.clip(output2save, 0 ,1)
                    plt.imshow(output2save)
                    plt.title(f'PSNR: {psnr}')
                    plt.tight_layout()
                    plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.png')
                    plt.close()
                    np.save(f'{save_name}/{model_name}_output_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',output2save)
                    torch.save({
                        'model_state_dict': grid.state_dict()
                    }, f'{save_name}/{model_name}_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                    torch.save({'model_state_dict': grid.state_dict()}, f'{save_name}/{model_name}_best_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                    
                else:
                    inr.train()
                    if batch_training:
                        tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} batch size: {num_batches} PSNR: 0.00')
                    else: 
                        tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} PSNR: 0.00')
                    for iter in tbar:
                        if batch_training:
                            perm = torch.randperm(num_samples)
                            x_shuffled = x[perm]
                            signal_shuffled = signal[perm]

                            for batch_idx in range(num_batches):
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, num_samples)
                                x_batch = x_shuffled[start_idx:end_idx].cuda()
                                signal_batch = signal_shuffled[start_idx:end_idx].cuda()
                                output = inr(x_batch).squeeze()
                                loss = loss_fn(output, signal_batch)
                                if TV_weight != 0:
                                    tv_loss = TV_Reg(inr.grid)
                                    loss += TV_weight[model_size_idx]*tv_loss
                                optim.zero_grad()
                                loss.backward()
                                optim.step()
                        else:   
                            if model == 'BACON':
                                output = inr(x)
                                losses = loss_fn_bacon(output, signal)
                                train_loss = 0.
                                for loss_name, loss in losses.items():
                                    single_loss = loss.mean()
                                    train_loss += single_loss
                                loss = train_loss
                            else:
                                output = inr(x).squeeze()
                                loss = loss_fn(output, signal)
                            if TV_weight != 0:
                                tv_loss = TV_Reg(inr.grid)
                                loss += TV_weight[model_size_idx]*tv_loss
                            optim.zero_grad()
                            loss.backward()
                            optim.step()   
                        
                        if iter % eval_step_size == 0 or iter == 1 or iter == num_steps:
                            with torch.no_grad():
                                inr.eval()
                                if not batch_training:
                                    # full batch
                                    if model == 'BACON':
                                        output = inr(x)[-1].squeeze()
                                    else:
                                        output = inr(x).squeeze()
                                    loss_val = loss_fn(output, signal).cpu().squeeze()
                                    psnr = -10*torch.log10(loss_val).numpy()
                                    model_psnrs.append(psnr)
                                    model_iters.append(iter+1)
                                    if iter == num_steps:
                                        output_eval = output

                                    tbar.set_description(f'max params {max_params:0.0e} PSNR: {psnr:.2f} best {best_psnr:.2f}')
                                    tbar.refresh() 
                                else:  # batch
                                    output_whole = torch.zeros_like(signal)
                                    for i, batch_idx in enumerate(range(num_batches)):
                                        
                                        start_idx = batch_idx * batch_size
                                        end_idx = min(start_idx + batch_size, num_samples)
                                        x_batch = x[start_idx:end_idx].cuda()
                                        signal_batch = signal[start_idx:end_idx].cuda()

                                        output_whole[start_idx:end_idx] = inr(x_batch).squeeze()
                                        
                                    loss_val = loss_fn(output_whole, signal).cpu().squeeze()
                                    psnr = -10*torch.log10(loss_val).numpy()

                                    tbar.set_description(f'max params {max_params:0.0e} batch size: {num_batches} PSNR: {psnr:.2f} best {best_psnr:.2f}')
                                    tbar.refresh() 

                                    model_psnrs.append(psnr)
                                    model_iters.append(iter+1)
                                    if iter == num_steps:
                                        output_eval = output_whole.reshape(signal_shape)
                                if psnr > best_psnr:
                                    best_psnr = psnr
                                    best_model_state = inr.state_dict()
                                    torch.save({'model_state_dict': best_model_state}, f'{save_name}/{model_name}_best_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                    output2save = output_eval.squeeze().detach().cpu().numpy()
                    if dimension == 2:
                        plt.figure()
                        output2save = np.clip(output2save, 0, 1)
                        plt.imshow(output2save)
                        plt.title(f'PSNR: {model_psnrs[-1]:.2f}')
                        plt.tight_layout()
                        plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.png')
                        plt.close()
                    np.save(f'{save_name}/{model_name}_output_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',output2save)

                    for iter, psnr in zip(model_iters, model_psnrs):
                        with open(iters_psnrs_name, 'a', newline='') as f:
                            writer_iter = csv.DictWriter(f, fieldnames=fieldnames_iters)
                            writer_iter.writerow({
                                'bandlimits': bandlimit, 
                                'num_params': max_params,
                                'iters': iter, 
                                'psnrs': f'{psnr:.2f}'
                            })
                    plt.figure()
                    plt.plot(model_iters, model_psnrs)
                    plt.ylim([0,50])
                    plt.title(f'best PSNR: {max(model_psnrs):.2f}')
                    plt.tight_layout()
                    plt.savefig(f'{save_name}/plots_{max_params:0.0e}_{bandlimit}_{selected_seed}.png')
                    plt.close()
                    
                # Append the new data to the CSV file in append mode ('a')
                    with open(csv_filename, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=['bandlimits', 'num_params', 'sigma', 'omega', 'grid', model_name])
                        writer.writerow({
                            'bandlimits': bandlimit,
                            'num_params': max_params,
                            'sigma': 0,
                            'omega': 0,
                            'grid': grid_psnr,
                            model_name: model_psnrs[-1]
                        })
                    torch.save({
                        'model_state_dict': inr.state_dict()
                    }, f'{save_name}/{model_name}_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                    
                    torch.cuda.empty_cache()

        if os.path.exists(save_name):    
            os.rename(save_name, save_name[:-1]+'_complete')
            tqdm.write(f"Folder renamed to: {save_name[:-1]+'_complete'}")
            tqdm.write(f"used {cuda_num} GPU for training")