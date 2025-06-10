import commentjson as json
import tinycudann as tcnn
import torch
import numpy as np
import torch.nn as nn
torch.set_default_dtype(torch.float)

class Instant_NGP(nn.Module):
	def __init__(self, dimension, max_params, out_features = 1, hidden_out = False, mapping_sigma = 20, num_hidden_layers = 2, mapping_size = 2000):
		super().__init__()
		self.out_features = out_features
		self.dimension = dimension
		with open("/home/namhoon/inr_Grid/git/tiny-cuda-nn/data/config_hash.json") as f:
			config = json.load(f)
		print(config)
		n_output_dims = 32
		num_neurons = config['network']['n_neurons']
		decoder_params = ((n_output_dims+1+out_features)*num_neurons + out_features)
		new_T = np.floor(np.log2((max_params - decoder_params)/32))
		config['encoding']['log2_hashmap_size'] = new_T # no hash collision when hash table is big

		hidden_layers = 2
		hidden_features = 64
        
		self.encoding = tcnn.Encoding(dimension, config["encoding"], dtype=torch.float)
		self.network = []
		for i in range(hidden_layers):
			if i == 0:
				self.network.append(nn.Linear(self.encoding.n_output_dims, hidden_features))
			elif i == hidden_layers - 1:
				self.network.append(nn.Linear(hidden_features, out_features))
			else:
				self.network.append(nn.Linear(hidden_features, hidden_features))
			if i < hidden_layers - 1:
				self.network.append(nn.ReLU())
		self.network = nn.Sequential(*self.network)

		self.model = torch.nn.Sequential(self.encoding, self.network).to(torch.float)
		print('model', self.model)
		self.shape = True
	def forward(self, x):
		shapes = list(x.shape)
		if len(x.shape) != 2:    
			x = x.reshape(-1, self.dimension)			
			self.shape = True
		else: self.shape = False
		if x.shape[-1] == self.dimension:
			pass
		x = self.encoding(x)
		x = self.network(x).to(torch.float)
		if self.shape == True:
			shapes[-1] = self.out_features
			x = x.reshape(shapes).squeeze()
		return x

def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
	dimension = 3
	for model_size in [1e4,3e4,1e5,3e5,1e6,3e6]:
		inr = Instant_NGP(dimension, max_params=model_size)

		print(inr, count_parameters(inr))
		print('')
