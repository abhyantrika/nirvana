# Based on https://github.com/lucidrains/siren-pytorch
from builtins import breakpoint
import torch
from torch import nn
from math import sqrt
import omegaconf
import math
import numpy as np

from .layers import ConvDecoder, DenseDecoder, Conv2d, Linear
from .bitEstimator import BitEstimator

from . import model_utils

class Sine(nn.Module):
	"""Sine activation with scaling.

	Args:
		w0 (float): Omega_0 parameter from SIREN paper.
	"""
	def __init__(self, w0=1.):
		super().__init__()
		self.w0 = w0

	def forward(self, x):
		return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
	"""Implements a single SIREN layer.

	Args:
		dim_in (int): Dimension of input.
		dim_out (int): Dimension of output.
		w0 (float):
		c (float): c value from SIREN paper used for weight initialization.
		is_first (bool): Whether this is first layer of model.
		use_bias (bool):
		activation (torch.nn.Module): Activation function. If None, defaults to
			Sine activation.
	"""
	def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False,
				 use_bias=True, activation=None,use_conv_mlp=False,weight_decoder=nn.Identity(),
                 bias_decoder=nn.Identity(), block_size=(1,1), name=''):
		super().__init__()
		self.dim_in = dim_in
		self.is_first = is_first
		self.use_conv_mlp = use_conv_mlp
		
		if self.use_conv_mlp:
			self.linear  = nn.Conv2d(dim_in, dim_out, kernel_size=(3,3),stride=(1,1),padding=(1,1),groups=1)
		else:	
			self.linear = Linear(dim_in, dim_out, weight_decoder, bias_decoder,
									bias=use_bias, name=name, block_size = block_size)			

		# Initialize layers following SIREN paper
		w_std = (1 / dim_in) if self.is_first else (sqrt(c / dim_in) / w0)
		nn.init.uniform_(self.linear.weight, -w_std, w_std)
		if use_bias:
			nn.init.uniform_(self.linear.bias, -w_std, w_std)

		self.activation = Sine(w0) if activation is None else activation

	def forward(self, x):
		out = self.linear(x)
		out = self.activation(out)
		return out

class PosEncoding(nn.Module):
	def __init__(self, dim, num_frames, freq):
		super().__init__()
		assert dim>1
		inv_freq = torch.zeros(dim)
		inv_freq[0::2] = torch.pow(1/freq, torch.arange(dim-dim//2))
		inv_freq[1::2] = torch.pow(1/freq, torch.arange(dim//2))
		pos_vec = inv_freq.unsqueeze(1)*torch.arange(num_frames).unsqueeze(0)
		pos_vec[1::2,:] += torch.pi/2
		self.pos_encoding = nn.Parameter(torch.sin(pos_vec).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),requires_grad=False)
		self.num_frames = num_frames
		assert self.pos_encoding.size() == (1,dim,num_frames,1,1)

	def forward(self, x):
		assert x.dim() == 4
		N, C, H, W = x.size()
		out = x.unsqueeze(2)+self.pos_encoding
		return out.reshape(N, C*self.num_frames, H, W)


class siren_nerv_optim(nn.Module):
	"""SIREN model with NERv blocks.
	
	Note: input has to be patches. 

	Args:
		dim_in (int): Dimension of input.
		dim_hidden (int): Dimension of hidden layers.
		dim_out (int): Dimension of output.
		num_layers (int): Number of layers.
		w0 (float): Omega 0 from SIREN paper.
		w0_initial (float): Omega 0 for first layer.
		use_bias (bool):
		final_activation (String): Activation function.
	"""
	def __init__(self, cfg):
		super().__init__()

		self.cfg = cfg
		self.nerv_cfg = self.cfg.network.nerv_config

		layers = []
		self.patch_out = omegaconf.OmegaConf.to_container(self.cfg.data.patch_shape)
		self.up_sample = self.nerv_cfg.up_sample #stride of upsampling
		self.dim_in = 2
		self.dim_out = 3
		self.use_bias = True
		self.group_size = self.cfg.trainer.group_size

		self.dim_hidden = self.cfg.network.layer_size
		self.num_layers = self.cfg.network.num_layers
		self.w0 = self.cfg.network.w0
		self.w0_initial = self.cfg.network.w0_initial
		
		self.nerv_config = self.cfg.network.nerv_config
		self.use_conv_mlp = self.cfg.network.use_conv_mlp

		self.final_activation = model_utils.get_activation(self.cfg.network.final_act)

		self.decoder_cfg = self.cfg.network.decoder_cfg


		# Create decoders, defaults to identity if not doing entropy based compression
		self.compress = True if  'entropy_reg' in  self.cfg.trainer.losses.loss_list else False

		weight_decoders, prob_models = {}, {}
		if self.decoder_cfg.decode_type == 'global':
			groups = {f'layer{k+1}': 'layer' for k in range(1, self.num_layers-1)}
			if self.decoder_cfg.compress_first:
				groups[f'layer1'] = (f'layer1' if self.decoder_cfg.unique_first else 'layer')
			else:
				groups['layer1'] = 'no_compress'
			groups[f'layer{self.num_layers}'] = (f'layer{self.num_layers}' 
													if self.decoder_cfg.unique_last else 'layer')
		else:
			start_idx = 0 if self.decoder_cfg.compress_first else 1
			groups = {f'layer{k+1}': f'layer{k+1}' for k in range(start_idx,self.num_layers)}
			if not self.decoder_cfg.compress_first:
				groups['layer1'] = 'no_compress'
		self.groups = groups
		
		self.unique_groups = list(set(groups.values()))
		conf_blocksize, conf_convdim = self.decoder_cfg.block_size, self.decoder_cfg.conv_dim
		conf_blocksize = {k:tuple(bs) for k,bs in conf_blocksize.items()}
		conf_convdim = {k:cd for k,cd in conf_convdim.items()}
		
		self.decoder_cfg = dict(self.decoder_cfg)
		del self.decoder_cfg['block_size']
		del self.decoder_cfg['conv_dim']
		self.decoder_cfg['std'] = 1.0
		
		for group in self.unique_groups:
			if group == 'no_compress':
				weight_decoders[group] = nn.Identity()
				prob_models[group] = nn.Identity()
				continue
			block_size = conf_blocksize.get(group, conf_blocksize['default'])
			conv_dim = conf_convdim.get(group,conf_convdim['default'])
			weight_decoders[group] = (ConvDecoder(block_size, conv_dim, **self.decoder_cfg) 
										if self.compress else nn.Identity())
			conv_dim = math.prod(block_size) if conv_dim == -1 else conv_dim
			prob_models[group] = BitEstimator(
								conv_dim if not self.cfg.network['single_prob_model'] else 1,
								self.cfg.network['symmetric_prob_model'], 
								num_layers=self.cfg.network['prob_num_layers']
								) if self.compress else nn.Identity()
			
		if self.compress:
			print('Unique groups: ', self.unique_groups)
			print('Block sizes:',{g:weight_decoders[self.groups[g]].block_size for g in self.groups if self.groups[g]!='no_compress'})
		
		self.weight_decoders = weight_decoders
		self.bias_decoder = nn.Identity() # todo: enable bias compression
		self.prob_models = prob_models


		for ind in range(self.num_layers-1):
			is_first = ind == 0
			layer_w0 = self.w0_initial if is_first else self.w0
			layer_dim_in = self.dim_in if is_first else self.dim_hidden
			group_name = f'layer{ind+1}'
			if self.compress and self.groups[group_name]!='no_compress':
				block_size = self.weight_decoders[self.groups[group_name]].block_size
			else:
				block_size = (1,1)

			layers.append(SirenLayer(
				dim_in=layer_dim_in,
				dim_out=self.dim_hidden,
				w0=layer_w0,
				use_bias=self.use_bias,
				is_first=is_first,
				use_conv_mlp=self.use_conv_mlp,
				weight_decoder = self.weight_decoders[self.groups[group_name]],
				bias_decoder = self.bias_decoder,
				block_size = block_size,
				name = group_name
			))
			print(group_name,self.weight_decoders[self.groups[group_name]],ind)

		self.expand_w = self.patch_out[0]//2
		self.expand_h = self.patch_out[1]//2
		self.expand_ch = self.nerv_cfg.expand_ch 
		self.expand_dims = self.expand_w * self.expand_h * self.expand_ch

		group_name = f'layer{self.num_layers}'
		if self.compress:
			block_size = self.weight_decoders[self.groups[group_name]].block_size

		layers.append(SirenLayer(dim_in=layer_dim_in, dim_out=self.expand_dims, w0=self.w0,
								use_bias=self.use_bias, activation=self.final_activation,\
								use_conv_mlp=self.use_conv_mlp,\
		                		name=group_name,weight_decoder=self.weight_decoders[self.groups[group_name]],
                				bias_decoder=self.bias_decoder,
                				block_size=block_size))
								

		self.stem = nn.Sequential(*layers)
		self.volume_out = self.group_size

		self.pre_nerv = PosEncoding(self.expand_ch, self.group_size, self.nerv_config.pos_enc_freq)
		self.nerv_in = self.expand_ch * self.group_size
		self.nerv_out = self.patch_out[1]*self.group_size
		self.last_in = self.patch_out[1]*self.group_size//(self.up_sample**2)
		self.num_groups = self.group_size

		self.nerv_block = nn.Sequential(nn.Conv2d(self.nerv_in,self.nerv_out, \
											kernel_size=(3,3),stride=(1,1),padding=(1,1),groups=self.num_groups),\
										nn.PixelShuffle(self.up_sample),\
										nn.GELU())

		self.last_layer = nn.Conv2d(self.last_in,self.volume_out*3,kernel_size=(3,3),\
							  stride=(1,1),padding=(1,1), groups=self.num_groups)

		if self.compress:
			boundaries = self.calc_boundaries() # calculate boundaries for each layer
			self.reset_parameters(boundaries)

	def calc_boundaries(self):
		boundaries = {}
		
		for m in self.modules():
			if isinstance(m,SirenLayer):
				group_name = self.groups[m.linear.name]				
				if group_name == 'no_compress':
					continue
				w0 = self.w0_initial if m.is_first else self.w0
				dim_in = m.dim_in
				w_std = (1 / dim_in) if m.is_first else (sqrt(6 / dim_in) / w0)
				if group_name not in boundaries:
					boundaries[group_name] = w_std
				else:
					boundaries[group_name] = min(boundaries[group_name],w_std)
		return boundaries

	def reset_parameters(self, boundaries):

		for group_name in self.weight_decoders:
			if group_name == 'no_compress':
				continue
			weight_decoder = self.weight_decoders[group_name]
			decode_matrix = self.decoder_cfg['decode_matrix']
			min_std = boundaries[group_name]
			boundary = self.decoder_cfg['boundary']
			if decode_matrix == 'sq':
				decoder_std = min_std/(weight_decoder.conv_dim*boundary)
			elif 'dft' in decode_matrix:
				decoder_std = min_std/(torch.mean(torch.sum(torch.abs(weight_decoder.dft),dim=0)).item()*boundary)
			decoder_std = 1/boundary
			if self.decoder_cfg['decode_norm']!='none':
				# decoder_std = 1
				decoder_std = min_std
			init_type = 'constant' if self.decoder_cfg['num_hidden'] == 0 else 'random'
			weight_decoder.reset_parameters(init_type,decoder_std/math.prod(weight_decoder.block_size))


		for m in self.modules():
			if isinstance(m,SirenLayer):
				if self.decoder_cfg['num_hidden']>0:
					continue
				group_name = self.groups[m.linear.name]
				if group_name == 'no_compress':
					continue
				weight_decoder = self.weight_decoders[group_name]
				decode_matrix = self.decoder_cfg['decode_matrix']
				w0 = self.w0_initial if m.is_first else self.w0
				dim_in = m.dim_in
				w_std = (1 / dim_in) if m.is_first else (sqrt(6 / dim_in) / w0)
				if decode_matrix == 'sq':
					boundary = w_std/torch.mean(torch.sum(torch.abs(weight_decoder.scale),dim=0))
				else:
					boundary = w_std/torch.mean(torch.sum(torch.abs(weight_decoder.dft),dim=0, keepdim=True)*
												torch.abs(weight_decoder.scale))
				boundary = torch.Tensor([1.0])
				mult = self.decoder_cfg['boundary_first'] if m.linear.name == 'layer1' else self.decoder_cfg['boundary']
				print(mult,m.linear.name)
				nn.init.uniform_(m.linear.weight, -mult, mult)


	def get_latents(self):
		weights = {}
		for m in self.modules():
			if isinstance(m, Linear):
				group_name = self.groups[m.name]
				if group_name == 'no_compress':
					continue
				weight = m.weight
				if group_name in weights:
					weights[group_name] = torch.cat((weights[group_name],weight))
				else:
					weights[group_name] = weight
	
		return weights



	def forward(self, x):
		if self.use_conv_mlp:
			x = x.view(x.size(0),x.size(1),1,1)
		out = self.stem(x)
		out = out.view(out.size(0),self.expand_ch,self.expand_h,self.expand_w)
		out = self.pre_nerv(out)
		out = self.nerv_block(out)
		out = self.last_layer(out)
		out = self.final_activation(out)		

		if self.cfg.data.data_format == 'patch_first':
			out = out.view(out.size(0),self.volume_out,3,self.patch_out[0],self.patch_out[1])
		else:
			out = out.view(self.volume_out,out.size(0),3,self.patch_out[0],self.patch_out[1])

		return out

