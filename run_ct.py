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
import utils
import matplotlib.cm as cm
from utils import *
import lpips
from skimage.metrics import structural_similarity as ssim
import pickle

os.environ['TORCH_CUDA_ARCH_LIST'] = "8.6"
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# colors = [cm.tab20(i) for i in range(20)]
seed_list = [1234, 2024, 5678, 7890, 7618]

selected_seed = seed_list[0]
cuda_num = 0
signal_num = -1
dimension = 2

set_seed(selected_seed)
sys.path.append('models/')
from utils import *
from models.inr import *

torch.set_default_dtype(torch.float)
torch.set_default_device(f'cuda:{cuda_num}')
torch.cuda.set_device(cuda_num)
device = f'cuda:{cuda_num}'
eval_step_size = 100
lpips_fn = lpips.LPIPS(net='vgg').to(device)
with open('hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)

models = ['FourFeat', 'Siren', 'Wire', 'GAPlane', 'Instant_NGP', 'GSplat', 'BACON', 'Grid', 'Grid_TV']
signals = [SparseSphereSignal, BandlimitedSignal, Sierpinski, StarTarget]
TV_weight = 0
TParams = {
    'signal_class': signals[signal_num],
    'dimension': dimension,
    'iters': 1000,
    'grid_interp': 'bicubic',
}
if signals[signal_num] in signals[2:]: # there are no randomness in those signals
    selected_seed = 1234

bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

if TParams['signal_class'].__name__ == 'StarTarget': 
    bandlimits = [0.1]

now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")

nmeas = 100
thetas = torch.tensor(np.linspace(0, 180, nmeas, dtype=np.float32)).cuda()

img = cv2.imread('target_signals/chest.png').astype(np.float32)[..., 1]
train_data = 'chest'

img = utils.normalize(img, True)
print(img.max(), img.min())
[H, W] = img.shape
CT_img = torch.tensor(img)[None, None, ...].cuda()

# Noise is not used in this script, but you can do so by modifying line 82 below
tau = 3e1                   # Photon noise (max. mean lambda). Set to 3e7 for representation, 3e1 for denoising
noise_snr = 2               # Readout noise (dB)
with torch.no_grad():
    sinogram = utils.radon(CT_img, thetas).detach().cpu()
    print(f'CT shape {CT_img.shape} sinogram shape: {sinogram.shape}')
    sinogram = sinogram.numpy()
    sinogram_noisy = utils.measure(sinogram,
                                    noise_snr,
                                    tau).astype(np.float32)
    # Set below to sinogram_noisy instead of sinogram to get noise in measurements
    
    signal = torch.tensor(sinogram).cuda()

trn_dataset, dataloader = init_dataloader(signal)
# print(signal.shape)
sH,sW = signal.shape
# hidden_layers = 2
# hidden_features = 156*np.sqrt(2)
batch_training = False
dimension = TParams['dimension']

if dimension == 2:
    visualize = True
    model_sizes = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
    # model_sizes = [3e6]
elif dimension == 3:
    model_sizes = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6]
    visualize = False
    TParams['grid_interp'] = 'bilinear'

# signal_class = TParams['signal_class']
signal_name = 'CT'
grid_class = get_model('Grid')

num_steps = TParams['iters']
interpolation_of_grid = TParams['grid_interp']

# signal = TParams['signal_class'](dimension, length = 1000, bandlimit = 0.1, seed = selected_seed)
# signal = torch.tensor(signal.signal, dtype=torch.float)
# print(f'signal shape : {H}, {W}')
coords_y = np.linspace(-1, 1, H, endpoint=False)
coords_x = np.linspace(-1, 1, W, endpoint=False)
x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y), -1), dtype=torch.float).cuda()
# print(x.shape, CT_img.shape)
num_samples = H*W#signal.shape[0]*signal.shape[1]
# print(num_samples)
# num_samples = len(signal.reshape(-1))
signal_shape = signal.shape
loss_fn = nn.MSELoss().cuda()
loss_fn_bacon = multiscale_image_mse
best_psnr = 0
total_psnrs = []
total_ssims = []
total_lpips = []
total_metrics = {}
model_bar = tqdm(models)
for model in model_bar:
    print('model name is', model)
    if model == 'BACON':
        coords_y = np.linspace(-0.5, 0.5, H, endpoint=False)
        coords_x = np.linspace(-0.5, 0.5, W, endpoint=False)
        print('different coordinates initialization for BACON is applied')
    else:
        coords_y = np.linspace(-1, 1, H, endpoint=False)
        coords_x = np.linspace(-1, 1, W, endpoint=False)
    x = torch.tensor(np.stack(np.meshgrid(coords_x, coords_y), -1), dtype=torch.float).cuda()
    if model == 'Grid_TV':
        model = 'Grid'
        TV_weight = [5e0, 1e1, 1e1, 1e1, 1e2, 1e2]
    # if model == 'Grid' or model == 'GAPlane':
    #     coords = np.linspace(-1, 1, signal.shape[0], endpoint=False)
    # else:
    #     coords = np.linspace(0, 1, signal.shape[0], endpoint=False)
    # x = torch.tensor(np.stack(np.meshgrid(*[coords for _ in range(dimension)]), -1), dtype=torch.float).cuda()
    if batch_training:
        x = x.reshape(-1, dimension)
    model_bar.set_description(f'{model}')
    model_class = get_model(model)
    model_name = model_class.__name__
    if TV_weight != 0:
        model_name = model+'_TV'
        print(model_name, TV_weight)
    # search_opt = hyperparams['models'][model]['hparams']
    learning_rate = hyperparams['models'][model]['lr']
    # learning_rate = 1e-4
    
            
    notes = f'{dimension}d_{signal_name}_{train_data}_{model_name}_lr_{learning_rate:0.0e}_{num_steps}_{selected_seed}'
    tqdm.write(notes)
    save_name = f'band_limit_figs/1CT/{formatted_time}_{notes}/'
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
    plt.figure()
    plt.imshow(sinogram)
    plt.title(f'Target')
    plt.tight_layout()
    plt.savefig(f'{save_name}/target_{selected_seed}.png')
    plt.close()
    max_params_bar = tqdm(model_sizes)
    model_outputs = []
    for model_size_idx, max_params in enumerate(max_params_bar):
        max_params_bar.set_description(f"{formatted_time} cuda {cuda_num} {selected_seed}")
        
        # grid_psnrs = torch.zeros((1,len(bandlimits)*len()*num_steps*10))
        
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
            
            if batch_training:
                signal = signal.reshape(num_samples)

            if model in ['GAPlane', 'GSplat', 'BACON']:
                inr = model_class(dimension, max_params, resolution = signal_shape).cuda()
            else:
                inr = model_class(dimension, max_params).cuda()
            optim = torch.optim.Adam(lr=learning_rate, params=inr.parameters())
            
            model_psnrs = []
            model_ssims = []
            model_lpips = []
            model_iters = []

            bandlim_bar.set_description(f"{dimension}D {signal_name} band limit {bandlimit}, lr={learning_rate:0.0e}")

            grid = grid_class(dimension=dimension, max_params=max_params, interpolation=interpolation_of_grid).cuda()
            optim_grid = torch.optim.Adam(lr=1e-1, params=[grid.grid])
            for iteration in tqdm(range(num_steps), desc='grid', leave=False):
                output = grid(x)
                output = output.reshape(1, 1, H,W)
                output = utils.radon(output, thetas)
                loss = loss_fn(output, signal)
                tv_loss = 0
                optim_grid.zero_grad()
                loss.backward()
                optim_grid.step()  
                if iteration == num_steps -1:
                    with torch.no_grad():
                        output = grid(x)
                        output = output.reshape(1, 1, H,W)
                        output = torch.clamp(output, min=0.0, max=1.0)
                        loss_val = loss_fn(output, CT_img).cpu().squeeze()
                        psnr = -10*torch.log10(loss_val).numpy()
                        grid_psnr = psnr
            torch.cuda.empty_cache()
            if model_name == 'Grid':
                output2save = output.reshape(H,W).squeeze().detach().cpu().numpy()
                plt.figure()
                output2save = np.clip(output2save, 0 ,1)
                model_outputs.append(output2save)
                plt.imshow(output2save)
                plt.title(f'PSNR: {psnr}')
                plt.tight_layout()
                plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.png')
                plt.close()
                np.save(f'{save_name}/{model}_output_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',output2save)
                torch.save({
                    'model_state_dict': grid.state_dict()
                }, f'{save_name}/Grid_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                torch.save({'model_state_dict': grid.state_dict()}, f'{save_name}/{model_name}_best_{max_params:0.0e}_{psnr:.2f}_{bandlimit}_{selected_seed}.pth')
                
            else:
                inr.train()
                if batch_training:
                    tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} batch size: {num_batches} PSNR: 0.00')
                else: 
                    tbar = tqdm(range(1,num_steps+1), desc=f'max params {max_params:0.0e} PSNR: 0.00')
                min_loss = float('inf')  # Very large initial value
                best_weights = None  # To store the best weights
                train_generator = iter(dataloader)
                for iteration in tbar:
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
                        if model != 'BACON':
                            output = inr(x).squeeze()
                            output = output.reshape(1, 1, H,W)
                            output = utils.radon(output, thetas)
                            loss = loss_fn(output, signal)
                        else: 
                            outputs = inr(x)
                            for i, output in enumerate(outputs):
                                outputs[i] = outputs[i].reshape(1, 1, H,W)
                                outputs[i] = utils.radon(outputs[i], thetas)
                            losses = loss_fn_bacon(outputs, signal)
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
                    
                    if iteration % eval_step_size == 0 or iteration == 1 or iteration == num_steps:
                        with torch.no_grad():
                            inr.eval()
                            if not batch_training:
                                # full batch
                                if model == 'BACON':
                                    output = inr(x)[-1].squeeze()
                                else:
                                    output = inr(x).squeeze()
                                output = output.reshape(1, 1, H,W)
                                output = torch.clamp(output, min=0.0, max=1.0)
                                cv2.imwrite(f'band_limit_figs/1CT/outputs_export/{model}_{max_params}.png', output.squeeze().cpu().detach().numpy()*255)
                                cv2.imwrite(f'band_limit_figs/1CT/outputs_export/{model}_{max_params}.jpeg', output.squeeze().cpu().detach().numpy()*255)
                                loss_val = loss_fn(output, CT_img).cpu().squeeze()
                                psnr = -10*torch.log10(loss_val).numpy()
                                model_psnrs.append(psnr)
                                model_iters.append(iteration+1)
                                if iteration == num_steps:
                                    output_eval = output
                                tbar.set_description(f'max params {max_params:0.0e} PSNR: {max(best_psnr,psnr):.2f}')
                                tbar.refresh() 
                            else:  # batch
                                output_whole = torch.zeros_like(signal)
                                for i, batch_idx in enumerate(range(num_batches)):
                                    
                                    start_idx = batch_idx * batch_size
                                    end_idx = min(start_idx + batch_size, num_samples)
                                    x_batch = x[start_idx:end_idx].cuda()
                                    signal_batch = signal[start_idx:end_idx].cuda()

                                    output_whole[start_idx:end_idx] = inr(x_batch).squeeze()
                                    output_whole = torch.clamp(output_whole, min=0.0, max=1.0)
                                    
                                loss_val = loss_fn(output_whole, CT_img).cpu().squeeze()
                                psnr = -10*torch.log10(loss_val).numpy()

                                tbar.set_description(f'max params {max_params:0.0e} batch size: {num_batches} PSNR: {max(best_psnr,psnr):.2f}')
                                tbar.refresh() 

                                model_psnrs.append(psnr)
                                model_iters.append(iteration+1)
                                if iteration == num_steps:
                                    output_eval = output_whole.reshape(H,W)
                            if loss_val < min_loss:
                                min_loss = loss_val  # Update the minimum loss
                                best_psnr = -10*torch.log10(min_loss).numpy()
                                best_weights = inr.state_dict()  # Save the current best model weights
                output2save = output_eval.squeeze().detach().cpu().numpy()
                model_outputs.append(output2save)
                if dimension == 2:
                    plt.figure()
                    plt.imshow(output2save, cmap='gray')
                    plt.title(f'{max_params:0.0e} PSNR: {model_psnrs[-1]:2}')
                    plt.tight_layout()
                    plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.png')
                    plt.savefig(f'{save_name}/output_{max_params:0.0e}_{bandlimit}_{selected_seed}.jpeg')
                    plt.close()
                np.save(f'{save_name}/{model}_output_{max_params:0.0e}_{bandlimit}_{selected_seed}.npy',output2save)

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
                plt.savefig(f'{save_name}/plots_{max_params:0.0e}_{bandlimit}_{selected_seed}.jpeg')
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
                    'model_state_dict': best_weights
                }, f'{save_name}/{model_name}_{max_params:0.0e}_{bandlimit}_{selected_seed}.pth')
                torch.cuda.empty_cache()
    plt.figure(figsize=(7,8))
    temp_psnrs = []
    temp_ssims = []
    temp_lpips = []
    for i, max_param in enumerate(model_sizes):
        plt.subplot(3,2,i+1)
        plt.imshow(model_outputs[i], cmap = 'gray')
        loss_val = np.mean(np.square(model_outputs[i]- CT_img.squeeze().cpu().numpy())).squeeze()
        ssim_val = ssim(model_outputs[i], CT_img.squeeze().cpu().numpy(), channel_axis=-1, data_range=CT_img.squeeze().cpu().numpy().max() - CT_img.squeeze().cpu().numpy().min())
        lpips_val = compute_lpips(model_outputs[i], CT_img.squeeze().cpu().numpy(), device)
        psnr = -10*np.log10(loss_val)
        temp_psnrs.append(psnr)
        temp_ssims.append(ssim_val)
        temp_lpips.append(lpips_val)
        plt.clim([0,1])
        plt.title(f'{max_param:0.0e} PSNR {psnr:.2f}')
    plt.suptitle(f'{model_name}', fontsize=24)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'{save_name}/{model_name}_{train_data}_CT.png')
    plt.savefig(f'{save_name}/{model_name}_{train_data}_CT.jpeg')
    plt.close()
    total_psnrs.append(temp_psnrs)
    total_ssims.append(temp_ssims)
    total_lpips.append(temp_lpips)
    total_metrics[model_name] = {'psnr': total_psnrs, 
                                  'ssim': total_ssims,
                                  'lpips':total_lpips}

    if os.path.exists(save_name):    
        os.rename(save_name, save_name[:-1]+'_complete')
        tqdm.write(f"Folder renamed to: {save_name[:-1]+'_complete'}")
        tqdm.write(f"used {cuda_num} GPU for training")

with open(f"band_limit_figs/1CT/CT_metrics.pkl", "wb") as f:
            pickle.dump(total_metrics, f)

psnrs2save = []
ssims2save = []
lpips2save = []
plt.figure()
model_sizes_str = ['1e+04', '3e+04', '1e+05', '3e+05', '1e+06', '3e+06']
for i, model_name in enumerate(models):
    if model_name == 'Grid_TV':
        model_name = 'Grid'
    plt.plot(total_psnrs[i], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name))
plt.title(f'CT psnrs')
plt.legend(loc = 'lower right')
plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes_str)
plt.savefig(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_plot.png')
plt.savefig(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_plot.jpeg')
np.save(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_plot.npy',total_psnrs)
plt.close()

psnrs2save = []
total_psnrs = np.array(total_psnrs)
plt.figure()
for i, model in enumerate(models):
    if model == 'Grid_TV':
        grid_idx = i
        break
for i, model_name in enumerate(models):
    if i == grid_idx:
        continue
    else:
        plt.plot(total_psnrs[i] - total_psnrs[grid_idx], label = plot_model_name(model_name), c = plot_colormap(model_name), linestyle = plot_linestyle(model_name))
        psnrs2save.append(total_psnrs[i] - total_psnrs[grid_idx])
plt.axhline(y=0, color='gray', linestyle='--', alpha = 0.2)
plt.title(f'CT psnrs')
plt.legend(loc = 'lower right')
plt.xticks(ticks=range(len(model_sizes)), labels=model_sizes_str)
plt.savefig(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_grid_plot.png')
plt.savefig(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_grid_plot.jpeg')
np.save(f'band_limit_figs/1CT/{formatted_time}_{train_data}_CT_grid_plot.npy',psnrs2save)
plt.close()
