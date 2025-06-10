import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import sys, os
sys.path.append('../')

from fourfeat import FourFeat
from grid import Grid
from siren import Siren 
from wire import Wire
from instant_ngp import Instant_NGP
from ga_plane import GAPlane
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gsplat', 'examples')))
from run_gsplat import GSplat
from bacon import BACON
torch.set_default_dtype(torch.float)

models_dict =  {'FourFeat': FourFeat,
                'Grid': Grid,
                'Siren': Siren,
                'Wire': Wire,
                'GAPlane' : GAPlane,
                'Instant_NGP' : Instant_NGP,
                'GSplat' : GSplat,
                'BACON' : BACON
                }


# maps the model 
def get_model(model_name):
    model_class = models_dict[model_name]
    return model_class

def train_INR(inr, signal, num_steps=100, eval_mode=False, evaluation_interval=100, batch_size=None):
    torch.cuda.empty_cache()
    
    # Construct the input indices based on the signal
    coords = np.linspace(0, 1, signal.shape[0], endpoint=False)
    x = torch.tensor(np.stack(np.meshgrid(*[coords for _ in range(inr.dimension)]), -1))
    
    # Flatten x and signal to make it easier to slice into mini-batches
    original_shape = signal.shape
    x = x.view(-1, inr.dimension)  # [num_samples, num_dimensions]
    signal = signal.view(-1)       # [num_samples]
    
    # Determine whether to use mini-batch or full-batch training
    num_samples = x.size(0)
    if batch_size is None:
        batch_size = num_samples  # Full-batch training if no batch size is provided
    num_batches = (num_samples + batch_size - 1) // batch_size  # Number of mini-batches

    evals = {'iters': [], 'outputs': []}
    
    for iters in tqdm(range(num_steps+1), position=4, leave=False, desc='train'):
        # Shuffle the dataset at the beginning of each iteration (optional)
        perm = torch.randperm(num_samples)
        x_shuffled = x[perm]
        signal_shuffled = signal[perm]
        
        for batch_idx in range(num_batches):
            # Get the mini-batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch = x_shuffled[start_idx:end_idx]
            signal_batch = signal_shuffled[start_idx:end_idx]

            # Apply the model to the mini-batch
            if inr.__class__.__name__ == 'Wire': 
                x_batch = x_batch.to(torch.float)
            output_batch = inr.model(x_batch).squeeze()
            if inr.__class__.__name__ == 'Wire':
                output_batch = output_batch.real

            # Compute the loss and take a step
            loss = inr.loss_fn(output_batch, signal_batch)
            inr.optimizer.zero_grad()
            loss.backward()
            inr.optimizer.step()
        torch.cuda.empty_cache()
        # Evaluation mode - save outputs and iterations
        if eval_mode and iters % evaluation_interval == 0:
            output_eval = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                x_batch = x[start_idx:end_idx]
                
                with torch.no_grad():
                    if inr.__class__.__name__ == 'Wire': 
                        x_batch = x_batch.to(torch.float)
                    output_batch = inr.model(x_batch).squeeze()
                    if inr.__class__.__name__ == 'Wire':
                        output_batch = output_batch.real
                    output_eval.append(output_batch)

            # Concatenate all mini-batch outputs to form the full evaluation output
            evals['outputs'].append(torch.cat(output_eval, dim=0).view(original_shape))
            evals['iters'].append(iters)

    if eval_mode:
        return evals
    else:
        # Perform final output using mini-batches
        output_final = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            x_batch = x[start_idx:end_idx]

            with torch.no_grad():
                if inr.__class__.__name__ == 'Wire': 
                    x_batch = x_batch.to(torch.float)
                output_batch = inr.model(x_batch).squeeze()
                if inr.__class__.__name__ == 'Wire':
                    output_batch = output_batch.real
                output_final.append(output_batch)

        return torch.cat(output_final, dim=0).view(original_shape)

if __name__ == '__main__':
    model_name = 'FourFeat'
    get_model(model_name)
    