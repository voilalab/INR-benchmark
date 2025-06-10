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
if len(sys.argv) > 1:
    var_seed, var_cuda, var_signal, var_dimension, var_sparse = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    selected_seed = seed_list[int(var_seed)]
    cuda_num = int(var_cuda)
    signal_num = int(var_signal)
    dimension = int(var_dimension)
    is_sparse = bool(int(var_sparse))
    tqdm.write(f'{selected_seed}')
else:
    selected_seed = seed_list[0]
    cuda_num = 6
    signal_num = 3
    dimension = 2
    is_sparse = False

note_sparse = ''
set_seed(selected_seed)

sys.path.append('models/')
from utils import *
from models.inr import *

torch.set_default_dtype(torch.float)
torch.cuda.set_device(cuda_num)

eval_step_size = 100

with open('hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)

if dimension == 3: # exclude GSplat
    models = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'BACON', 'Grid']
else:
    models = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'BACON', 'GSplat', 'BACON', 'Grid']

signals = [SparseSphereSignal, BandlimitedSignal, Sierpinski, StarTarget, Voxel_Fitting]

TParams = {
    'signal_class': signals[signal_num],
    'dimension': dimension,
    'iters': 1000,
    'grid_interp': 'bicubic',
}
if signals[signal_num] in signals[2:]: # there are no randomness in those signals
    selected_seed = 1234

bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
if TParams['signal_class'].__name__ == 'StarTarget' or TParams['signal_class'].__name__ == 'Voxel_Fitting': 
    bandlimits = [0.1]

now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")

batch_training = True
dimension = TParams['dimension']

if dimension == 2:
    visualize = True
    model_sizes = [1e4]
elif dimension == 3:
    model_sizes = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
    visualize = False
    TParams['grid_interp'] = 'bilinear'

signal_class = TParams['signal_class']
signal_name = signal_class.__name__
grid_class = get_model('Grid')

num_steps = TParams['iters']
interpolation_of_grid = TParams['grid_interp']

signal = TParams['signal_class'](dimension, length = 1000, bandlimit = 0.1, seed = selected_seed)
signal = torch.tensor(signal.signal, dtype=torch.float)

num_samples = len(signal.reshape(-1))
signal_shape = signal.shape

loss_fn_bacon = multiscale_image_mse
loss_fn = nn.MSELoss().cuda()

TV_weight = 0
model_bar = tqdm(models)
for model in model_bar:
    if model == 'Grid_TV':
        model = 'Grid'
        TV_weight = [1e-3, 1e-3, 1e-2, 1e-1, 1e-1, 1e-1]
    # fake initialization
    signal = TParams['signal_class'](dimension, length = 1000, bandlimit = 0.1, seed = selected_seed, sparse = is_sparse)
    signal = torch.tensor(signal.signal, dtype=torch.float).cuda()
    signal_shape = signal.shape
    
    if model == 'BACON':
            coord_limit = 0.5
    else: coord_limit = 1

    if signal_name == 'Voxel_Fitting':
        coords_y = np.linspace(-coord_limit, coord_limit, signal.shape[0], endpoint=False)
        coords_x = np.linspace(-coord_limit, coord_limit, signal.shape[1], endpoint=False)
        coords_z = np.linspace(-coord_limit, coord_limit, signal.shape[2], endpoint=False)
        x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y, coords_z), -1), dtype=torch.float).cuda()
        if is_sparse: note_sparse = '_sparse'
    else:
        coords = np.linspace(-coord_limit, coord_limit, signal.shape[0], endpoint=False)
        x = torch.tensor(np.stack(np.meshgrid(*[coords for _ in range(dimension)]), -1), dtype=torch.float).cuda()

    if batch_training:
        x = x.reshape(-1, dimension)
    model_bar.set_description(f'{model}')
    model_class = get_model(model)
    model_name = model_class.__name__
    if TV_weight != 0:
        model_name = model+'_TV'
    learning_rate = hyperparams['models'][model]['lr']    
            
    notes = f'{dimension}d_{signal_name}_{model_name}_lr_{learning_rate:0.0e}_{num_steps}_{selected_seed}'
    tqdm.write(notes)
    save_name = f'band_limit_figs/{dimension}d_{signal_name}{note_sparse}/{formatted_time}_{notes}/'
    target_dir = make_folders(save_name)
    # If the CSV file doesn't exist, create it and write the header
    csv_filename = save_name + 'records.csv'
    iters_psnrs_name = save_name + 'iters_psnrs.csv'
    times_name = save_name + 'times.csv'
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
    if not os.path.exists(times_name):
        with open(times_name, 'w', newline='') as f:
            fieldnames_time = ['bandlimits', 'num_params', 'iters']
            writer_time = csv.DictWriter(f, fieldnames=fieldnames_time)
            writer_time.writeheader()
    max_params_bar = tqdm(model_sizes)
    time_list = {}
    time_list['max_params'] = []
    time_list['iters'] = []
    for model_size_idx, max_params in enumerate(max_params_bar):
        max_params_bar.set_description(f"{formatted_time} cuda {cuda_num} {selected_seed}")
        
        bandlim_bar = tqdm(bandlimits)

        if batch_training:
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
            
            # real initialization for the signal
            signal = TParams['signal_class'](dimension, length = 1000, bandlimit = bandlimit, seed = selected_seed, sparse = is_sparse)
            signal = torch.tensor(signal.signal, dtype=torch.float).cuda()
            trn_dataset, dataloader = init_dataloader(signal, 1, signal_shape)
            
            if batch_training:
                signal = signal.reshape(num_samples)

            if model in ['GAPlane', 'GSplat', 'BACON']:
                inr = model_class(dimension, max_params, resolution = signal_shape).cuda()
            else:
                inr = model_class(dimension, max_params).cuda()
            if 'FourFeat' in model:
                np.save(f'{save_name}/{model}_BMat_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',inr.B.cpu().numpy())
            optim = torch.optim.Adam(lr=learning_rate, params=inr.parameters())

            model_psnrs = []
            model_iters = []

            bandlim_bar.set_description(f"{dimension}D {signal_name} band limit {bandlimit}, lr={learning_rate:0.0e}")

            grid = grid_class(dimension=dimension, max_params=max_params, interpolation=interpolation_of_grid).cuda()
            optim_grid = torch.optim.Adam(lr=0.1, params=[grid.grid])
            for iteration in tqdm(range(num_steps), desc='grid', leave=False):
                output = grid(x)
                if not batch_training:
                    output = output.reshape(signal_shape)
                loss = loss_fn(output, signal)
                optim_grid.zero_grad()
                loss.backward()
                optim_grid.step()  
                if iteration == num_steps -1:
                    with torch.no_grad():
                        output = grid(x)
                        if not batch_training:
                            output = output.reshape(signal_shape)
                        loss = loss_fn(output, signal)
                        loss_val = loss_fn(output, signal).cpu().squeeze()
                        psnr = -10*torch.log10(loss_val).numpy()
                        grid_psnr = psnr
            torch.cuda.empty_cache()

            inr.train()
            if batch_training:
                tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} batch size: {num_batches} PSNR: 0.00')
            else: 
                tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} PSNR: 0.00')
            temp_time_list = []
            train_generator = iter(dataloader)
            init_time = time.time()
            for iteration in tbar:
                init_iter_time = time.time()
                if batch_training:
                    perm = torch.randperm(num_samples)
                    x_shuffled = x[perm]
                    signal_shuffled = signal[perm]
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, num_samples)
                        x_batch = x_shuffled[start_idx:end_idx].cuda()
                        signal_batch = signal_shuffled[start_idx:end_idx].cuda()
                        
                        if model != 'BACON':
                            output = inr(x_batch).squeeze()
                            loss = loss_fn(output, signal_batch)
                        else: # BACON has different loss function
                            output = inr(x_batch)
                            losses = loss_fn_bacon(output, signal_batch.unsqueeze(-1))
                            train_loss = 0.
                            for loss_name, loss in losses.items():
                                single_loss = loss.mean()
                                train_loss += single_loss
                            loss = train_loss
                        if TV_weight != 0:
                            tv_loss = TV_Reg(inr.grid)
                            loss += TV_weight[model_size_idx]*tv_loss
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                else:   
                    try:
                        _x, _signal = next(train_generator)
                    except StopIteration:
                        train_generator = iter(dataloader)
                        _x, _signal = next(train_generator)
                    _x, _signal = dict2cuda(_x), dict2cuda(_signal)
                    _output = inr((x/2).reshape(1,-1,2))
                    losses = loss_fn(_output, _signal)
                    if TV_weight != 0:
                        tv_loss = TV_Reg(inr.grid)
                        loss += TV_weight[model_size_idx]*tv_loss
                    
                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        train_loss += single_loss
                    loss = train_loss

                    optim.zero_grad()
                    loss.backward()
                    optim.step()   
                temp_time_list.append(time.time()-init_iter_time)
                if iteration % eval_step_size == 0 or iteration == 1 or iteration == num_steps:
                    with torch.no_grad():
                        inr.eval()
                        if not batch_training:
                            # full batch
                            output = inr(x)
                            output = output['model_out']['output'][0].reshape(signal_shape).squeeze()
                            output = torch.clamp(output, min=0.0, max=1.0)
                            output_whole = output
                            loss_val = loss_fn(output_whole, signal).cpu().squeeze()
                            psnr = -10*torch.log10(loss_val.cpu().detach()).numpy()
                            model_psnrs.append(psnr)
                            model_iters.append(iteration+1)
                            if iteration == num_steps:
                                output_eval = output_whole = output
                            tbar.set_description(f'max params {max_params:0.0e} PSNR: {max(psnr, best_psnr):.2f}')
                            tbar.refresh() 
                        else:  # batch
                            output_whole = torch.zeros_like(signal)
                            for i, batch_idx in enumerate(range(num_batches)):
                                
                                start_idx = batch_idx * batch_size
                                end_idx = min(start_idx + batch_size, num_samples)
                                x_batch = x[start_idx:end_idx].cuda()
                                signal_batch = signal[start_idx:end_idx].cuda()
                                if model == 'BACON':
                                    output_whole[start_idx:end_idx] = inr(x_batch)[-1].squeeze()
                                else:
                                    output_whole[start_idx:end_idx] = inr(x_batch).squeeze()
                                output_whole = torch.clamp(output_whole, min=0.0, max=1.0)
                                
                            loss_val = loss_fn(output_whole, signal).cpu().squeeze()
                            psnr = -10*torch.log10(loss_val).numpy()

                            tbar.set_description(f'max params {max_params:0.0e} batch size: {num_batches} PSNR: {psnr:.2f}, {best_psnr:.2f}')
                            tbar.refresh() 
                            model_psnrs.append(psnr)
                            model_iters.append(iteration+1)
                            if iteration == num_steps:
                                output_eval = output_whole.reshape(signal_shape)
                        if psnr > best_psnr:
                            tqdm.write(f'psnr {psnr:.2f},best {best_psnr:.2f}')
                            best_psnr = psnr
                            best_model_state = inr.state_dict()
                            np.save(f'{save_name}/{model}_output_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy', output_whole.view(signal_shape).cpu().numpy())
                            plt.figure(2)
                            img = output_whole.view(signal_shape).cpu().numpy()
                            if dimension == 3:
                                img = img[:,:,len(img[0,0])//2]
                            plt.imshow(img)
                            plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.jpeg')
                            torch.save({
                                'model_state_dict': best_model_state
                            }, f'{save_name}/{model_name}_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                            plt.close(2)


            time_list['iters'].append(temp_time_list)
            
            final_time = time.time() - init_time
            tqdm.write(f'{np.mean(temp_time_list)} {final_time}')
            time_list['max_params'].append(final_time)
            with open(times_name, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_time)
                writer.writerow({
                    'bandlimits': final_time,
                    'num_params': final_time,
                    'iters': np.mean(temp_time_list)
                })
            for iteration, psnr in zip(model_iters, model_psnrs):
                with open(iters_psnrs_name, 'a', newline='') as f:
                    writer_iter = csv.DictWriter(f, fieldnames=fieldnames_iters)
                    writer_iter.writerow({
                        'bandlimits': bandlimit, 
                        'num_params': max_params,
                        'iters': iteration, 
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
            torch.cuda.empty_cache()
    if os.path.exists(save_name):    
        os.rename(save_name, save_name[:-1]+'_complete')
        tqdm.write(f"Folder renamed to: {save_name[:-1]+'_complete'}")
        tqdm.write(f"used {cuda_num} GPU for training")
